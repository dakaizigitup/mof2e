"""
PaiNN + MOF Global Feature Injection
按照 eSCN 方案：FiLM 只作用于 scalar x [N, hidden_channels]，
vector vec [N, 3, hidden_channels] 绝不修改（保持等变性）。

注入设计：
  Layer 0  : nm FiLM + atom FiLM（context_ln → film_gamma/beta → x）
  Layer K  : global FiLM（MOFGlobalEncoderV2 → global_gamma/beta → x）
             inject BEFORE layer_blocks[K]（遵循 eSCN 约定）

与 eSCN 关键对应：
  eSCN  x.embedding[:, 0, :]  ←→  PaiNN  x  (scalar, [N, hidden_channels])
  eSCN  x.embedding[:, 1:, :] ←→  PaiNN  vec (不动)
  eSCN  context_ln + film_gamma/beta  ←→  PaiNN 完全一致
  eSCN  global_gamma_mlp (Linear→GELU→Linear)  ←→  PaiNN 完全一致

两阶段训练：
  Stage 1：freeze_first_n_layers=0，全量训练 nm + atom
  Stage 2：freeze_first_n_layers=N，冻结前 N 层 + nm/atom 模块，训练 global FiLM + 后 N 层
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch_scatter import scatter

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import HeadInterface
from fairchem.core.models.escn.escn import ConditionEncoder

from .painn import PaiNN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_last(seq: nn.Sequential) -> None:
    """Zero-init the last Linear in a Sequential (adapter_zero_init)."""
    for m in reversed(list(seq)):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            return


# ---------------------------------------------------------------------------
# Backbone: PaiNNGlobal
# ---------------------------------------------------------------------------

@registry.register_model("painn_global")
class PaiNNGlobal(PaiNN):
    """PaiNN backbone + MOF condition/global FiLM injection.

    Strictly follows eSCN injection philosophy:
      - FiLM operates on scalar x [N, C] ONLY
      - vec [N, 3, C] is NEVER modified (equivariance preserved)
      - context_ln normalises [nm || atom] before FiLM (eSCN style)
      - global FiLM injected BEFORE layer global_inject_layer (eSCN convention)
      - global_layer_norm: optional LayerNorm on global encoder output

    Two-stage training:
      Stage 1: freeze_first_n_layers=0 → train everything
      Stage 2: freeze_first_n_layers=N → freeze emb + layers[0:N] + nm/atom FiLM,
               train global FiLM + layers[N:] + energy head.
    """

    def __init__(
        self,
        # --- Standard PaiNN params ---
        hidden_channels: int = 512,
        num_layers: int = 6,
        num_rbf: int = 128,
        cutoff: float = 12.0,
        max_neighbors: int = 50,
        rbf: dict | None = None,
        envelope: dict | None = None,
        regress_forces: bool = False,
        direct_forces: bool = False,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        otf_graph: bool = True,
        num_elements: int = 83,
        scale_file: str | None = None,
        # --- nm condition ---
        base_nm_dim: int = 2,
        condition_hidden_dim: int = 64,
        nm_max_count: float = 15.0,
        # --- atom extra features ---
        use_atom_extra_features: bool = False,
        atom_extra_dim: int = 128,
        atom_encoder_type: str = "v2",
        # --- MOF global encoder ---
        use_mof_global_features: bool = False,
        mof_global_excel_path: str | None = None,
        mof_global_dim: int = 64,
        mof_global_encoder_type: str = "v2",
        mof_smi_ted_folder: str | None = None,
        mof_selfies_ted_path: str | None = None,
        mof_mhg_path: str | None = None,
        mof_smiles_cache_path: str | None = None,
        # --- FiLM adapter ---
        adapter_hidden_dim: int = 128,
        adapter_dropout: float = 0.0,
        adapter_zero_init: bool = True,
        # --- Injection position ---
        global_inject_layer: int = 3,
        # --- Injection mode：
        #     "film" = x*(1+γ) + β  (默认，与之前 checkpoint 兼容)
        #     "gated" = x + sigmoid(gate) ⊙ residual  (门控残差，绕开 LayerNorm 抹除)
        global_inject_mode: str = "film",
        # --- Normalization on global encoder output ---
        global_layer_norm: bool = False,
        # --- Two-stage freeze ---
        freeze_first_n_layers: int = 0,
    ) -> None:
        super().__init__(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            rbf=rbf,
            envelope=envelope,
            regress_forces=regress_forces,
            direct_forces=direct_forces,
            use_pbc=use_pbc,
            use_pbc_single=use_pbc_single,
            otf_graph=otf_graph,
            num_elements=num_elements,
            scale_file=scale_file,
        )

        # Nullify original energy head (replaced by WeightedEnergyHead)
        self.out_energy = None

        # Save config
        self.base_nm_dim = base_nm_dim
        self.use_atom_extra_features = bool(use_atom_extra_features)
        self.atom_extra_dim = int(atom_extra_dim)
        self.use_mof_global_features = bool(use_mof_global_features)
        self.mof_global_dim = int(mof_global_dim)
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.global_inject_layer = int(global_inject_layer)
        self.global_layer_norm = bool(global_layer_norm)
        self.freeze_first_n_layers = int(freeze_first_n_layers)

        # ── ConditionEncoder (nm → condition_hidden_dim) ─────────────────────
        self.condition_encoder = ConditionEncoder(
            input_dim=base_nm_dim,
            num_gaussians=20,
            out_dim=condition_hidden_dim,
            hidden_dim=adapter_hidden_dim,
            dropout=adapter_dropout,
            nm_max_count=nm_max_count,
        )
        self.nm_encoded_dim: int = condition_hidden_dim

        # ── AtomPropertyEncoderV2 ────────────────────────────────────────────
        self.atom_encoder = None
        if self.use_atom_extra_features:
            try:
                if atom_encoder_type.lower() == "v2":
                    from fairchem.core.models.equiformer_v2.atomic_emb_v2 import (
                        AtomPropertyEncoderV2,
                    )
                    self.atom_encoder = AtomPropertyEncoderV2(
                        max_Z=num_elements, out_dim=atom_extra_dim
                    )
                else:
                    from fairchem.core.models.equiformer_v2.atomic_emb import (
                        AtomPropertyEncoder,
                    )
                    self.atom_encoder = AtomPropertyEncoder(
                        max_Z=num_elements, out_dim=atom_extra_dim
                    )
                logging.info(f"[PaiNNGlobal] AtomPropertyEncoder ({atom_encoder_type}) OK, out_dim={atom_extra_dim}")
            except Exception as e:
                logging.warning(f"[PaiNNGlobal] AtomPropertyEncoder init failed: {e}")
                self.atom_encoder = None

        # context = nm_encoded [N, nm_encoded_dim] + atom_feat [N, atom_extra_dim]
        self.context_input_dim: int = self.nm_encoded_dim
        if self.use_atom_extra_features and self.atom_encoder is not None:
            self.context_input_dim += self.atom_extra_dim

        # ── MOFGlobalEncoderV2 ───────────────────────────────────────────────
        self.mof_global_encoder = None
        if self.use_mof_global_features and mof_global_excel_path:
            _dev = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                from fairchem.core.models.equiformer_v2.global_emb_v2 import (
                    MOFGlobalEncoderV2,
                )
                self.mof_global_encoder = MOFGlobalEncoderV2(
                    excel_path=mof_global_excel_path,
                    smi_ted_folder=mof_smi_ted_folder,
                    selfies_ted_path=mof_selfies_ted_path,
                    mhg_path=mof_mhg_path,
                    global_dim=mof_global_dim,
                    device=_dev,
                    cache_path=mof_smiles_cache_path,
                )
                logging.info("[PaiNNGlobal] MOFGlobalEncoderV2 initialised")
            except Exception as e:
                logging.warning(f"[PaiNNGlobal] MOFGlobalEncoderV2 init failed: {e}")

        # ── Context FiLM (Layer 0) ── eSCN style ─────────────────────────────
        # context_ln: LayerNorm on [nm || atom] (Protection 1, 对齐 eSCN)
        self.context_ln = nn.LayerNorm(self.context_input_dim)
        # film_gamma/beta: Linear → LayerNorm → GELU → Dropout → Linear
        # 严格对齐 eSCN escn.py 中 film_gamma 的结构（带 hidden LN + Dropout）
        self.film_gamma = nn.Sequential(
            nn.Linear(self.context_input_dim, adapter_hidden_dim),
            nn.LayerNorm(adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(adapter_dropout),
            nn.Linear(adapter_hidden_dim, hidden_channels),
        )
        self.film_beta = nn.Sequential(
            nn.Linear(self.context_input_dim, adapter_hidden_dim),
            nn.LayerNorm(adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(adapter_dropout),
            nn.Linear(adapter_hidden_dim, hidden_channels),
        )
        if adapter_zero_init:
            _zero_last(self.film_gamma)
            _zero_last(self.film_beta)

        # ── Global injection modules ──────────────────────────────────────────
        # mode="film"：γ, β 两个 MLP → x = x*(1+γ) + β  (默认，旧 checkpoint 兼容)
        # mode="gated"：gate, residual 两个 MLP → x = x + sigmoid(gate) ⊙ residual
        #               门控残差：避开 LayerNorm 对 scale-shift 信号的抹除
        self.global_inject_mode = global_inject_mode.lower().strip()
        assert self.global_inject_mode in ("film", "gated"), \
            f"global_inject_mode must be 'film' or 'gated', got {global_inject_mode}"

        self.global_gamma_mlp: nn.Sequential | None = None   # FiLM gamma
        self.global_beta_mlp: nn.Sequential | None = None    # FiLM beta
        self.global_gate_mlp: nn.Sequential | None = None    # Gated: sigmoid gate
        self.global_residual_mlp: nn.Sequential | None = None # Gated: residual
        self.global_input_norm: nn.LayerNorm | None = None
        self.global_hidden_norm: nn.LayerNorm | None = None

        if self.use_mof_global_features:
            # global_layer_norm 保留作为可选的 input norm（作用在 global encoder 输出上）
            # 同时对齐 GemNet 方案：在 adapter hidden 上再做一次归一化，稳定全局注入幅度
            if self.global_layer_norm:
                self.global_input_norm = nn.LayerNorm(mof_global_dim)
                self.global_hidden_norm = nn.LayerNorm(adapter_hidden_dim)

            if self.global_inject_mode == "film":
                self.global_gamma_mlp = nn.Sequential(
                    nn.Linear(mof_global_dim, adapter_hidden_dim),
                    self.global_hidden_norm if self.global_layer_norm else nn.Identity(),
                    nn.GELU(),
                    nn.Linear(adapter_hidden_dim, hidden_channels),
                )
                self.global_beta_mlp = nn.Sequential(
                    nn.Linear(mof_global_dim, adapter_hidden_dim),
                    nn.LayerNorm(adapter_hidden_dim) if self.global_layer_norm else nn.Identity(),
                    nn.GELU(),
                    nn.Linear(adapter_hidden_dim, hidden_channels),
                )
                if adapter_zero_init:
                    _zero_last(self.global_gamma_mlp)
                    _zero_last(self.global_beta_mlp)
                logging.info("[PaiNNGlobal] global_gamma_mlp / global_beta_mlp (eSCN 简化版) init OK, zero-init")
            else:  # "gated"
                # residual: h(global), zero-init → 初始 h=0 → x=x（恒等）
                # 加入和 GemNet 一致的 hidden LayerNorm，稳定 global adapter 尺度
                self.global_residual_mlp = nn.Sequential(
                    nn.Linear(mof_global_dim, adapter_hidden_dim),
                    nn.LayerNorm(adapter_hidden_dim) if self.global_layer_norm else nn.Identity(),
                    nn.GELU(),
                    nn.Linear(adapter_hidden_dim, hidden_channels),
                )
                # gate: 不 zero-init（sigmoid(0)=0.5，让 gate 有中等开度，方便学）
                self.global_gate_mlp = nn.Sequential(
                    nn.Linear(mof_global_dim, adapter_hidden_dim),
                    nn.LayerNorm(adapter_hidden_dim) if self.global_layer_norm else nn.Identity(),
                    nn.GELU(),
                    nn.Linear(adapter_hidden_dim, hidden_channels),
                )
                if adapter_zero_init:
                    # 只 zero residual，保证初始 x=x 恒等变换
                    _zero_last(self.global_residual_mlp)
                logging.info("[PaiNNGlobal] global_inject_mode = gated (residual zero-init, eSCN 简化版)")

        # ── Two-stage freeze ─────────────────────────────────────────────────
        if freeze_first_n_layers > 0:
            n = min(freeze_first_n_layers, num_layers)
            self.atom_emb.requires_grad_(False)
            self.radial_basis.requires_grad_(False)
            for i in range(n):
                self.message_layers[i].requires_grad_(False)
                self.update_layers[i].requires_grad_(False)
                getattr(self, f"upd_out_scalar_scale_{i}").requires_grad_(False)
            # nm/atom FiLM modules (trained in stage1, frozen in stage2)
            self.condition_encoder.requires_grad_(False)
            self.film_gamma.requires_grad_(False)
            self.film_beta.requires_grad_(False)
            self.context_ln.requires_grad_(False)
            if self.atom_encoder is not None:
                self.atom_encoder.requires_grad_(False)
            logging.info(
                f"[PaiNNGlobal] Frozen: atom_emb + radial_basis + "
                f"message/update_layers[0:{n}] + ScaleFactors[0:{n}] + "
                f"condition_encoder + film_gamma/beta + context_ln + atom_encoder"
            )

    # ── Forward helpers ──────────────────────────────────────────────────────

    def _get_nm_per_node(self, data, batch, num_atoms, device):
        if hasattr(data, "condition") and data.condition is not None:
            cond = data.condition
            if isinstance(cond, torch.Tensor):
                cond = cond.to(device=device, dtype=torch.float32)
                if cond.dim() == 1:
                    cond = cond.view(1, -1)
            else:
                cond = torch.tensor(cond, device=device, dtype=torch.float32).view(1, -1)

            if cond.size(0) == 1 and cond.size(1) > self.base_nm_dim:
                n_cond = cond.size(1) // self.base_nm_dim
                if cond.size(1) % self.base_nm_dim == 0 and n_cond == len(data.natoms):
                    cond = cond.view(n_cond, self.base_nm_dim)

            if cond.size(0) == 1 and len(data.natoms) > 1:
                cond = cond.expand(len(data.natoms), -1)

            nm_encoded = self.condition_encoder(cond)
            return nm_encoded[batch]
        else:
            return torch.zeros(num_atoms, self.nm_encoded_dim, device=device, dtype=torch.float32)

    def _get_global_per_node(self, data, batch, num_atoms, device):
        if self.mof_global_encoder is None:
            return None
        if not (hasattr(data, "name") and data.name is not None):
            return None

        mof_embeddings = []
        names = data.name if isinstance(data.name, (list, tuple)) else [data.name]
        for name in names:
            name = str(name)
            mof_name = name.split("_")[0] if "_" in name else name
            mof_embeddings.append(self.mof_global_encoder(mof_name))
        while len(mof_embeddings) < len(data.natoms):
            mof_embeddings.append(mof_embeddings[-1])

        global_emb = torch.cat(mof_embeddings, dim=0).to(device=device)
        global_emb = torch.nan_to_num(global_emb, nan=0.0, posinf=0.0, neginf=0.0)
        return global_emb[batch]

    # ── Main forward ─────────────────────────────────────────────────────────

    @conditional_grad(torch.enable_grad())
    def forward(self, data) -> dict[str, torch.Tensor]:
        """PaiNN forward with nm + global FiLM injection on scalar x.

        Key: vec [N, 3, C] is NEVER touched — equivariance preserved.
        nm/atom FiLM at Layer 0, global FiLM BEFORE layer global_inject_layer.
        """
        pos = data.pos
        batch = data.batch
        z = data.atomic_numbers.long()
        device = pos.device
        num_atoms = len(z)

        (
            edge_index,
            neighbors,
            edge_dist,
            edge_vector,
            id_swap,
        ) = self.generate_graph_values(data)

        edge_rbf = self.radial_basis(edge_dist)

        x = self.atom_emb(z)                              # [N, hidden_channels]
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        # ── Layer 0: nm FiLM + atom FiLM on x (eSCN approach) ────────────────
        nm_per_node = self._get_nm_per_node(data, batch, num_atoms, device)
        context_parts = [nm_per_node]
        if self.use_atom_extra_features and self.atom_encoder is not None:
            tags = data.tags if hasattr(data, "tags") else None
            atom_feat = self.atom_encoder(z, tags=tags).to(device=device)
            context_parts.append(atom_feat)
        context = torch.cat(context_parts, dim=-1)        # [N, context_input_dim]
        context = self.context_ln(context)                 # eSCN: LN before FiLM
        gamma = self.film_gamma(context)                   # [N, hidden_channels]
        beta  = self.film_beta(context)
        x = x * (1.0 + gamma) + beta
        # vec NOT modified

        # ── Global embedding lookup ───────────────────────────────────────────
        global_per_node = None
        has_global_mlp = (
            (self.global_gamma_mlp is not None and self.global_beta_mlp is not None)
            or (self.global_gate_mlp is not None and self.global_residual_mlp is not None)
        )
        if self.mof_global_encoder is not None and has_global_mlp:
            global_per_node = self._get_global_per_node(data, batch, num_atoms, device)
            if global_per_node is not None and self.global_input_norm is not None:
                global_per_node = self.global_input_norm(global_per_node)

        # ── Helper: 根据 mode 执行一次 global 注入 ────────────────────────────
        def _apply_global(x_in):
            if self.global_inject_mode == "film":
                gamma_g = self.global_gamma_mlp(global_per_node)
                beta_g  = self.global_beta_mlp(global_per_node)
                return x_in * (1.0 + gamma_g) + beta_g
            else:  # "gated"
                gate = torch.sigmoid(self.global_gate_mlp(global_per_node))
                residual = self.global_residual_mlp(global_per_node)
                return x_in + gate * residual

        # ── Interaction blocks ────────────────────────────────────────────────
        for i in range(self.num_layers):
            # Mid-loop 注入 BEFORE layer i (0 < inject_layer < num_layers)
            # 注意：PaiNNMessage 内部有 x_layernorm，mid-loop 的 FiLM/gated 信号
            # 会被部分吸收。如果希望 global 影响最大化，建议 post-loop 注入。
            if (
                i == self.global_inject_layer
                and 0 < self.global_inject_layer < self.num_layers
                and global_per_node is not None
            ):
                x = _apply_global(x)
                # vec NOT modified

            dx, dvec = self.message_layers[i](x, vec, edge_index, edge_rbf, edge_vector)
            x   = x   + dx
            vec = vec + dvec
            x   = x   * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)
            x   = x   + dx
            vec = vec + dvec
            x   = getattr(self, "upd_out_scalar_scale_%d" % i)(x)

        # ── Post-loop 注入 (inject_layer >= num_layers) ───────────────────────
        # 在所有 layer 之后注入，global 信息直接影响最终 x，不被任何 LayerNorm 抹除。
        if (
            self.global_inject_layer >= self.num_layers
            and global_per_node is not None
        ):
            x = _apply_global(x)

        return {"node_features": x, "node_vec": vec, "batch": batch}

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Weighted Energy Head
# ---------------------------------------------------------------------------

@registry.register_model("painn_global_weighted_energy_head")
class PaiNNGlobalWeightedEnergyHead(nn.Module, HeadInterface):
    """PhAST-style weighted energy head for PaiNNGlobal.

    E_total = scatter(E_i * w_i, batch)
    where E_i = energy_mlp(x_i),  w_i = sigmoid(weight_mlp(x_i)).
    """

    def __init__(
        self,
        backbone,
        reduce: str = "weighted_sum",
        weight_nn_hidden_dim: int = 64,
        freeze_head: bool = False,
        **kwargs,
    ):
        super().__init__()

        if isinstance(backbone, dict):
            bb_cfg = dict(backbone)
            bb_name = bb_cfg.pop("name", "painn_global")
            backbone = registry.get_model_class(bb_name)(**bb_cfg)

        self.backbone = backbone
        self.reduce = reduce
        C = backbone.hidden_channels

        self.energy_mlp = nn.Sequential(
            nn.Linear(C, weight_nn_hidden_dim),
            nn.SiLU(),
            nn.Linear(weight_nn_hidden_dim, weight_nn_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(weight_nn_hidden_dim // 2, 1),
        )

        if "weighted" in reduce:
            self.weight_mlp = nn.Sequential(
                nn.Linear(C, weight_nn_hidden_dim),
                nn.SiLU(),
                nn.Linear(weight_nn_hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.weight_mlp = None

        # ── Freeze energy head (stage2 纯 global 注入模式)─────────────────
        # 当 backbone 全冻结 + head 全冻结时，只有 global FiLM 可训练。
        # 这样保证 stage2 不会比 stage1 差：γ=β=0 时 = stage1。
        if freeze_head:
            self.energy_mlp.requires_grad_(False)
            if self.weight_mlp is not None:
                self.weight_mlp.requires_grad_(False)
            logging.info("[PaiNNGlobalWeightedEnergyHead] Frozen: energy_mlp + weight_mlp")

    def forward(
        self,
        data,
        emb: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if emb is None:
            emb = self.backbone(data)

        x     = emb["node_features"]          # [N, hidden_channels]
        batch = emb.get("batch", data.batch)
        num_systems = len(data.natoms)

        node_energy = self.energy_mlp(x)      # [N, 1]

        if self.weight_mlp is not None:
            node_weights = self.weight_mlp(x) # [N, 1]
            weighted_node_energy = node_energy * node_weights
        else:
            weighted_node_energy = node_energy
            node_weights = torch.ones_like(node_energy)

        energy = scatter(
            weighted_node_energy.view(-1), batch,
            dim=0, dim_size=num_systems, reduce="add",
        )

        if self.reduce in ("weighted_sum_normalized", "mean"):
            weight_sums = scatter(
                node_weights.view(-1), batch,
                dim=0, dim_size=num_systems, reduce="add",
            )
            weight_sums = torch.clamp(weight_sums, min=1e-6)
            energy = energy / weight_sums

        return {"energy": energy}
