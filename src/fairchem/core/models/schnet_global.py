"""
SchNet + MOF Global Feature Injection
MOF backbone comparison experiment (PhAST backbone suite).

Backbone: SchNet (E(3)-invariant, scalar h [N, hidden_channels])
Injection 设计（按信号语义分层）:
  Layer 0   : nm FiLM 条件注入
              gamma_nm, beta_nm = MLP(nm_encoded)
              h = h * (1 + gamma_nm) + beta_nm
  Layer 0   : atom FiLM 注入（与 nm 对称，独立，不影响 nm 信号）
              gamma_a, beta_a = MLP(atom_feat)  [zero-init last layer]
              h = h * (1 + gamma_a) + beta_a
  Layer K   : global FiLM 注入（MOFGlobalEncoderV2）
              gamma_g, beta_g = MLP(global_emb)
              h = h * (1 + gamma_g) + beta_g

所有注入模块均 zero-init → 初始等价于无注入，训练稳定。
nm 仅注入第 0 层，atom 仅注入第 0 层，global 在 global_inject_layer 处注入。

Usage (yaml):
  model:
    name: schnet_global_weighted_energy_head
    backbone:
      name: schnet_global
      ...
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch_geometric.nn import SchNet

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import GraphModelMixin, HeadInterface
from fairchem.core.models.escn.escn import ConditionEncoder


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

@registry.register_model("schnet_global")
class SchNetGlobal(SchNet, GraphModelMixin):
    """SchNet backbone with MOF condition + global feature injection.

    Injection strategy:
      - Layer 0  : nm FiLM — h = h*(1+γ_nm) + β_nm   [zero-init → identity at init]
      - Layer 0  : atom additive — h = h + atom_proj(atom_feat)  [zero-init]
      - Layer K  : global FiLM — h = h*(1+γ_g) + β_g  where K = global_inject_layer
    All injection modules are zero-initialised so training is stable from the start.

    Two-stage training:
      Stage 1: freeze_first_n_layers=0  →  train everything (baseline)
      Stage 2: freeze_first_n_layers=N  →  freeze embedding + interactions[0:N],
               train global FiLM + interactions[N:] + energy head.
    """

    def __init__(
        self,
        # --- PyG SchNet params ---
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        regress_forces: bool = False,
        otf_graph: bool = False,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 8.0,
        readout: str = "sum",
        max_num_elements: int = 100,
        # --- Condition (nm) encoder ---
        base_nm_dim: int = 2,
        condition_hidden_dim: int = 64,
        nm_max_count: float = 15.0,
        # --- Atom property encoder ---
        use_atom_extra_features: bool = True,
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
        # --- Normalization on global encoder output before FiLM ---
        global_layer_norm: bool = False,
        # --- Two-stage freeze ---
        freeze_first_n_layers: int = 0,
    ) -> None:
        # Attributes expected by GraphModelMixin.generate_graph
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.otf_graph = otf_graph
        self.max_neighbors = 50
        self.cutoff = cutoff

        # PyG SchNet.__init__ creates: embedding, interactions, distance_expansion,
        # lin1, lin2, act (ShiftedSoftplus)
        super().__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )

        # Save config
        self.hidden_channels = hidden_channels
        self.base_nm_dim = base_nm_dim
        self.use_atom_extra_features = bool(use_atom_extra_features)
        self.atom_extra_dim = int(atom_extra_dim)
        self.use_mof_global_features = bool(use_mof_global_features)
        self.mof_global_dim = int(mof_global_dim)
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.adapter_dropout = float(adapter_dropout)
        self.adapter_zero_init = bool(adapter_zero_init)
        self.global_inject_layer = int(global_inject_layer)
        self.global_layer_norm = bool(global_layer_norm)
        self.freeze_first_n_layers = int(freeze_first_n_layers)
        self.max_num_elements = int(max_num_elements)

        # ── ConditionEncoder (nm → 64-dim) ──────────────────────────────────
        self.condition_encoder = ConditionEncoder(
            input_dim=base_nm_dim,
            num_gaussians=20,
            out_dim=condition_hidden_dim,
            hidden_dim=adapter_hidden_dim,
            dropout=adapter_dropout,
            nm_max_count=nm_max_count,
        )
        self.nm_encoded_dim: int = condition_hidden_dim

        # ── AtomPropertyEncoder ─────────────────────────────────────────────
        self.atom_encoder = None
        if self.use_atom_extra_features:
            try:
                if atom_encoder_type.lower() == "v2":
                    from fairchem.core.models.equiformer_v2.atomic_emb_v2 import (
                        AtomPropertyEncoderV2,
                    )
                    self.atom_encoder = AtomPropertyEncoderV2(
                        max_Z=max_num_elements, out_dim=atom_extra_dim
                    )
                else:
                    from fairchem.core.models.equiformer_v2.atomic_emb import (
                        AtomPropertyEncoder,
                    )
                    self.atom_encoder = AtomPropertyEncoder(
                        max_Z=max_num_elements, out_dim=atom_extra_dim
                    )
                logging.info(f"AtomPropertyEncoder ({atom_encoder_type}) initialised")
            except Exception as e:
                logging.warning(f"AtomPropertyEncoder init failed: {e}")
                self.atom_encoder = None

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
                logging.info("MOFGlobalEncoderV2 (三模态) initialised")
            except Exception as e:
                logging.warning(f"MOFGlobalEncoderV2 init failed: {e}")

        def _zero_last(seq: nn.Sequential) -> None:
            for m in reversed(list(seq)):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    return

        # ── Layer-0: nm FiLM 条件注入 ────────────────────────────────────────
        # nm 语义是"条件调制"（T/P 改变模型对结构的响应），FiLM 比 concat 更合适。
        # h = h * (1 + gamma_nm) + beta_nm
        # MLP: Linear(nm_encoded_dim, adapter_hidden_dim) → GELU → Linear → hidden_channels
        # zero-init last layer → 初始 gamma=0, beta=0，等价于恒等变换
        self.nm_gamma_mlp = nn.Sequential(
            nn.Linear(self.nm_encoded_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, hidden_channels),
        )
        self.nm_beta_mlp = nn.Sequential(
            nn.Linear(self.nm_encoded_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, hidden_channels),
        )
        if adapter_zero_init:
            _zero_last(self.nm_gamma_mlp)
            _zero_last(self.nm_beta_mlp)

        # ── Layer-0: atom 加法残差注入（独立，不影响 nm 信号）──────────────
        # atom_feat 语义是"表示增强"（补充物理属性），加法残差即可。
        # delta = atom_proj(atom_feat)，zero-init → 初始无影响
        # ── Layer-0: atom FiLM 注入（与 nm 注入对称）────────────────────────
        # atom_feat 语义是"表示增强"，FiLM 比单层加法残差更有表达力。
        # h = h * (1 + gamma_a) + beta_a
        self.atom_gamma_mlp: nn.Sequential | None = None
        self.atom_beta_mlp:  nn.Sequential | None = None
        if self.atom_encoder is not None:
            self.atom_gamma_mlp = nn.Sequential(
                nn.Linear(atom_extra_dim, adapter_hidden_dim),
                nn.GELU(),
                nn.Linear(adapter_hidden_dim, hidden_channels),
            )
            self.atom_beta_mlp = nn.Sequential(
                nn.Linear(atom_extra_dim, adapter_hidden_dim),
                nn.GELU(),
                nn.Linear(adapter_hidden_dim, hidden_channels),
            )
            if adapter_zero_init:
                _zero_last(self.atom_gamma_mlp)
                _zero_last(self.atom_beta_mlp)

        # ── 中间层全局 FiLM ──────────────────────────────────────────────────
        # h = h * (1 + gamma_g) + beta_g，在 global_inject_layer 处注入
        self.global_gamma_mlp: nn.Sequential | None = None
        self.global_beta_mlp:  nn.Sequential | None = None
        self.global_input_norm: nn.LayerNorm | None = None
        if self.use_mof_global_features:
            # optional LayerNorm on encoder output (decouples pretrained scale)
            if self.global_layer_norm:
                self.global_input_norm = nn.LayerNorm(mof_global_dim)
            self.global_gamma_mlp = nn.Sequential(
                nn.Linear(mof_global_dim, adapter_hidden_dim),
                nn.LayerNorm(adapter_hidden_dim) if self.global_layer_norm else nn.Identity(),
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

        # ── Two-stage freeze ─────────────────────────────────────────────────
        # Stage2 冻结：embedding + interactions[0:n] + nm/atom FiLM 模块
        # 保留已学好的局部表示，只训练全局 FiLM
        if freeze_first_n_layers > 0:
            # backbone embedding & interactions
            self.embedding.requires_grad_(False)
            n = min(freeze_first_n_layers, len(self.interactions))
            for i in range(n):
                self.interactions[i].requires_grad_(False)
            # nm 编码器 + FiLM（Stage1 训练好，Stage2 不动）
            self.condition_encoder.requires_grad_(False)
            self.nm_gamma_mlp.requires_grad_(False)
            self.nm_beta_mlp.requires_grad_(False)
            # atom 编码器 + FiLM（Stage1 训练好，Stage2 不动）
            if self.atom_encoder is not None:
                self.atom_encoder.requires_grad_(False)
            if self.atom_gamma_mlp is not None:
                self.atom_gamma_mlp.requires_grad_(False)
                self.atom_beta_mlp.requires_grad_(False)
            logging.info(
                f"[SchNetGlobal] Frozen: embedding + interactions[0:{n}] "
                f"+ condition_encoder + nm_film + atom_encoder + atom_film"
            )

    # ── Forward helpers ──────────────────────────────────────────────────────

    def _get_global_per_node(self, data, batch, num_atoms, device):
        """Look up MOF global embeddings and expand to per-node [N, mof_global_dim]."""
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
        # Pad to batch size if needed
        while len(mof_embeddings) < len(data.natoms):
            mof_embeddings.append(mof_embeddings[-1])

        global_emb = torch.cat(mof_embeddings, dim=0).to(device=device)  # [B, G]
        global_emb = torch.nan_to_num(global_emb, nan=0.0, posinf=0.0, neginf=0.0)
        return global_emb[batch]  # [N, G]

    def _get_nm_per_node(self, data, batch, num_atoms, device):
        """Encode nm condition and expand to per-node [N, nm_encoded_dim]."""
        if hasattr(data, "condition") and data.condition is not None:
            cond = data.condition
            if isinstance(cond, torch.Tensor):
                cond = cond.to(device=device, dtype=torch.float32)
                if cond.dim() == 1:
                    cond = cond.view(1, -1)
            else:
                cond = torch.tensor(cond, device=device, dtype=torch.float32).view(1, -1)

            # Handle batched condition: [1, B*base_nm_dim] → [B, base_nm_dim]
            if cond.size(0) == 1 and cond.size(1) > self.base_nm_dim:
                n_cond = cond.size(1) // self.base_nm_dim
                if cond.size(1) % self.base_nm_dim == 0 and n_cond == len(data.natoms):
                    cond = cond.view(n_cond, self.base_nm_dim)

            if cond.size(0) == 1 and len(data.natoms) > 1:
                cond = cond.expand(len(data.natoms), -1)

            nm_encoded = self.condition_encoder(cond)   # [B, nm_encoded_dim]
            return nm_encoded[batch]                    # [N, nm_encoded_dim]
        else:
            return torch.zeros(num_atoms, self.nm_encoded_dim, device=device, dtype=torch.float32)

    # ── Main forward ─────────────────────────────────────────────────────────

    def forward_embeddings(self, data) -> dict[str, torch.Tensor]:
        """Returns {'node_features': h [N, hidden_channels]} after all interactions."""
        device = data.pos.device
        z = data.atomic_numbers.long()
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
                torch.zeros(len(z), dtype=torch.long, device=device)
        num_atoms = len(z)

        graph = self.generate_graph(data)
        edge_attr = self.distance_expansion(graph.edge_distance)

        h = self.embedding(z)   # [N, hidden_channels]

        # ── Prepare global per-node (needed before interaction loop) ─────────
        global_per_node = self._get_global_per_node(data, batch, num_atoms, device)
        if global_per_node is not None and self.global_input_norm is not None:
            global_per_node = self.global_input_norm(global_per_node)

        # ── Layer-0: nm FiLM 条件注入 ────────────────────────────────────────
        nm_per_node = self._get_nm_per_node(data, batch, num_atoms, device)
        gamma_nm = self.nm_gamma_mlp(nm_per_node)   # [N, hidden_channels]
        beta_nm  = self.nm_beta_mlp(nm_per_node)    # [N, hidden_channels]
        h = h * (1.0 + gamma_nm) + beta_nm

        # ── Layer-0: atom FiLM 注入（独立） ─────────────────────────────────
        if self.atom_encoder is not None and self.atom_gamma_mlp is not None:
            tags = data.tags if hasattr(data, "tags") else None
            atom_feat = self.atom_encoder(z, tags=tags).to(device=device)
            gamma_a = self.atom_gamma_mlp(atom_feat)   # [N, hidden_channels]
            beta_a  = self.atom_beta_mlp(atom_feat)    # [N, hidden_channels]
            h = h * (1.0 + gamma_a) + beta_a

        # ── Interaction blocks ───────────────────────────────────────────────
        for i, interaction in enumerate(self.interactions):
            # Mid-layer global FiLM (inject BEFORE interaction i)
            if (
                i == self.global_inject_layer
                and 0 < self.global_inject_layer < len(self.interactions)
                and global_per_node is not None
                and self.global_gamma_mlp is not None
            ):
                gamma_g = self.global_gamma_mlp(global_per_node)
                beta_g  = self.global_beta_mlp(global_per_node)
                h = h * (1.0 + gamma_g) + beta_g

            h = h + interaction(h, graph.edge_index, graph.edge_distance, edge_attr)

        # Post-loop global FiLM (global_inject_layer >= num_interactions)
        if (
            self.global_inject_layer >= len(self.interactions)
            and global_per_node is not None
            and self.global_gamma_mlp is not None
        ):
            gamma_g = self.global_gamma_mlp(global_per_node)
            beta_g  = self.global_beta_mlp(global_per_node)
            h = h * (1.0 + gamma_g) + beta_g

        return {"node_features": h}

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        """Standalone forward (uses PyG lin1/lin2 energy head)."""
        emb = self.forward_embeddings(data)
        h = emb["node_features"]
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
                torch.zeros_like(data.atomic_numbers.long())

        h2 = self.lin1(h)
        h2 = self.act(h2)
        h2 = self.lin2(h2)

        energy = h2.new_zeros(batch.max().item() + 1).scatter_reduce_(
            0, batch, h2[:, 0], reduce=self.reduce, include_self=False
        )[:, None]
        return energy

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)
        outputs = {"energy": energy}
        if self.regress_forces:
            forces = -1 * torch.autograd.grad(
                energy,
                data.pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
            outputs["forces"] = forces
        return outputs

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Energy Head
# ---------------------------------------------------------------------------

@registry.register_model("schnet_global_weighted_energy_head")
class SchNetGlobalWeightedEnergyHead(nn.Module, HeadInterface):
    """PhAST-style weighted energy head for SchNetGlobal.

    E_total = Σ_i  w_i · e_i
    where e_i = energy_mlp(h_i),  w_i = sigmoid(weight_mlp(h_i)).

    Mirrors eSCNWeightedEnergyHead but operates on scalar h [N, C]
    instead of sphere_values [N*num_samples, C_all].
    """

    def __init__(
        self,
        backbone,
        reduce: str = "weighted_sum",
        weight_nn_hidden_dim: int = 64,
        use_initial_embeddings: bool = True,
        condition_dim: int = 2,
        condition_hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__()

        # Instantiate backbone from dict if needed
        if isinstance(backbone, dict):
            bb_cfg = dict(backbone)
            bb_name = bb_cfg.pop("name", "schnet_global")
            backbone = registry.get_model_class(bb_name)(**bb_cfg)

        self.backbone = backbone
        self.reduce = reduce
        self.use_initial_embeddings = use_initial_embeddings

        C = backbone.hidden_channels

        # Energy MLP: h → node_energy [N, 1]
        self.energy_mlp = nn.Sequential(
            nn.Linear(C, weight_nn_hidden_dim),
            nn.SiLU(),
            nn.Linear(weight_nn_hidden_dim, weight_nn_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(weight_nn_hidden_dim // 2, 1),
        )

        # Weight MLP: h → node_weight ∈ [0, 1]
        if "weighted" in reduce:
            self.weight_mlp = nn.Sequential(
                nn.Linear(C, weight_nn_hidden_dim),
                nn.SiLU(),
                nn.Linear(weight_nn_hidden_dim, 1),
                nn.Sigmoid(),
            )
            if use_initial_embeddings:
                self.weight_mlp_init = nn.Sequential(
                    nn.Linear(C, weight_nn_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(weight_nn_hidden_dim, 1),
                    nn.Sigmoid(),
                )
            else:
                self.weight_mlp_init = None
        else:
            self.weight_mlp = None
            self.weight_mlp_init = None

    def forward(
        self,
        data,
        emb: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if emb is None:
            emb = self.backbone.forward_embeddings(data)

        h = emb["node_features"]   # [N, hidden_channels]

        node_energy = self.energy_mlp(h)   # [N, 1]

        if self.weight_mlp is not None:
            node_weights = self.weight_mlp(h)
            if self.weight_mlp_init is not None:
                node_weights = node_weights + self.weight_mlp_init(h)
            weighted_node_energy = node_energy * node_weights
        else:
            weighted_node_energy = node_energy
            node_weights = torch.ones_like(node_energy)

        num_systems = len(data.natoms)
        energy = torch.zeros(num_systems, device=data.pos.device)
        energy.index_add_(0, data.batch, weighted_node_energy.view(-1))

        if self.reduce in ("weighted_sum_normalized", "mean"):
            weight_sums = torch.zeros(num_systems, device=data.pos.device)
            weight_sums.index_add_(0, data.batch, node_weights.view(-1))
            weight_sums = torch.clamp(weight_sums, min=1e-6)
            energy = energy / weight_sums
        elif self.reduce in ("weighted_sum", "sum"):
            pass
        else:
            raise ValueError(f"Unknown reduce: {self.reduce}")

        return {"energy": energy}
