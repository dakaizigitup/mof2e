"""
DimeNet++ + MOF Global Feature Injection
MOF backbone comparison experiment (PhAST backbone suite).

Backbone: DimeNet++ (directional, angle-aware, 3-body interactions)
Injection 设计（按信号语义分层）:
  Layer 0   : nm FiLM 条件注入
              h = h * (1 + gamma_nm) + beta_nm
  Layer 0   : atom FiLM 注入（与 nm 对称，独立）
              h = h * (1 + gamma_a) + beta_a
  Layer K   : global FiLM 注入（MOFGlobalEncoderV2）
              h = h * (1 + gamma_g) + beta_g

注入目标：x [N, hidden_channels=128]（节点特征，类比 eSCN 的 l0 monopole）
能量预测：PhAST 风格加权 energy head（与 SchNet 版本对称）
所有注入模块均 zero-init → 初始等价于无注入，训练稳定。
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch_scatter import scatter

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import GraphModelMixin, HeadInterface
from fairchem.core.models.dimenet_plus_plus import DimeNetPlusPlusWrap
from fairchem.core.models.escn.escn import ConditionEncoder


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

@registry.register_model("dimenetplusplus_global")
class DimeNetPlusPlusGlobal(DimeNetPlusPlusWrap):
    """DimeNet++ backbone with MOF condition + global feature injection.

    Injection strategy (all zero-init → identity at init):
      - Layer 0  : nm FiLM — h = h*(1+γ_nm) + β_nm
      - Layer 0  : atom FiLM — h = h*(1+γ_a) + β_a
      - Layer K  : global FiLM — h = h*(1+γ_g) + β_g  where K = global_inject_layer

    The injection target is x [N, hidden_channels], analogous to eSCN's l0 monopole.
    Energy is computed by PhAST-style weighted energy head on final x.

    Two-stage training:
      Stage 1: freeze_first_n_layers=0  →  train everything
      Stage 2: freeze_first_n_layers=N  →  freeze emb + interaction_blocks[0:N],
               train global FiLM + interaction_blocks[N:] + energy head.
    """

    def __init__(
        self,
        # --- DimeNet++ params ---
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        regress_forces: bool = False,
        otf_graph: bool = False,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        int_emb_size: int = 64,
        basis_emb_size: int = 8,
        out_emb_channels: int = 256,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = 10.0,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
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
        global_inject_layer: int = 2,
        # --- Two-stage freeze ---
        freeze_first_n_layers: int = 0,
    ) -> None:
        super().__init__(
            use_pbc=use_pbc,
            use_pbc_single=use_pbc_single,
            regress_forces=regress_forces,
            otf_graph=otf_graph,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
        )

        # Save config
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.base_nm_dim = base_nm_dim
        self.use_atom_extra_features = bool(use_atom_extra_features)
        self.atom_extra_dim = int(atom_extra_dim)
        self.use_mof_global_features = bool(use_mof_global_features)
        self.mof_global_dim = int(mof_global_dim)
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.global_inject_layer = int(global_inject_layer)
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

        # ── nm FiLM (Layer 0) ────────────────────────────────────────────────
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

        # ── atom FiLM (Layer 0, 独立) ────────────────────────────────────────
        self.atom_gamma_mlp: nn.Sequential | None = None
        self.atom_beta_mlp: nn.Sequential | None = None
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

        # ── 中间层 / 后置全局 FiLM ───────────────────────────────────────────
        self.global_gamma_mlp: nn.Sequential | None = None
        self.global_beta_mlp: nn.Sequential | None = None
        if self.use_mof_global_features:
            self.global_gamma_mlp = nn.Sequential(
                nn.Linear(mof_global_dim, adapter_hidden_dim),
                nn.GELU(),
                nn.Linear(adapter_hidden_dim, hidden_channels),
            )
            self.global_beta_mlp = nn.Sequential(
                nn.Linear(mof_global_dim, adapter_hidden_dim),
                nn.GELU(),
                nn.Linear(adapter_hidden_dim, hidden_channels),
            )
            if adapter_zero_init:
                _zero_last(self.global_gamma_mlp)
                _zero_last(self.global_beta_mlp)

        # ── Two-stage freeze ─────────────────────────────────────────────────
        if freeze_first_n_layers > 0:
            # backbone basis & embedding
            self.rbf.requires_grad_(False)
            self.sbf.requires_grad_(False)
            self.emb.requires_grad_(False)
            # interaction blocks
            n = min(freeze_first_n_layers, len(self.interaction_blocks))
            for i in range(n):
                self.interaction_blocks[i].requires_grad_(False)
            # nm 编码器 + FiLM
            self.condition_encoder.requires_grad_(False)
            self.nm_gamma_mlp.requires_grad_(False)
            self.nm_beta_mlp.requires_grad_(False)
            # atom 编码器 + FiLM
            if self.atom_encoder is not None:
                self.atom_encoder.requires_grad_(False)
            if self.atom_gamma_mlp is not None:
                self.atom_gamma_mlp.requires_grad_(False)
                self.atom_beta_mlp.requires_grad_(False)
            logging.info(
                f"[DimeNetPlusPlusGlobal] Frozen: rbf + sbf + emb + "
                f"interaction_blocks[0:{n}] + condition_encoder + nm_film + "
                f"atom_encoder + atom_film"
            )

    # ── Forward helpers ──────────────────────────────────────────────────────

    def _get_global_per_node(self, data, batch, device):
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
        return global_emb[batch]  # [N, G]

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

    # ── Main forward ─────────────────────────────────────────────────────────

    @conditional_grad(torch.enable_grad())
    def forward_embeddings(self, data) -> dict[str, torch.Tensor]:
        """DimeNet++ forward with nm FiLM injection on edge features.

        DimeNet++ 的核心特征是边特征 x [E, hidden_channels]，而不是节点特征。
        nm 条件通过 FiLM 直接调制边特征（emb 之后、interaction_blocks 之前），
        这样 nm 信号会随消息传递扩散到整个图的每条边。

        数据流：
          emb(z, rbf, i, j) → x [E, C]
          nm FiLM: x = x * (1 + γ) + β  ← nm 注入点（边级）
          output_blocks[0](x) → P [N, 1]
          for each interaction_block + output_block:
              x = interaction_block(x)   ← nm 信号随消息传递扩散
              P += output_block(x)       → [N, 1]
          scatter(P, batch) → 系统能量
        """
        pos = data.pos
        device = pos.device
        z = data.atomic_numbers.long()
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
                torch.zeros(len(z), dtype=torch.long, device=device)
        num_atoms = len(z)

        graph = self.generate_graph(data)
        data.edge_index = graph.edge_index
        data.cell_offsets = graph.cell_offsets
        data.neighbors = graph.neighbors
        j, i = graph.edge_index

        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            graph.edge_index,
            data.cell_offsets,
            num_nodes=num_atoms,
        )

        # Angles
        pos_i = pos[idx_i].detach()
        pos_j = pos[idx_j].detach()
        if self.use_pbc:
            pos_ji = pos[idx_j].detach() - pos_i + graph.offset_distances[idx_ji]
            pos_kj = pos[idx_k].detach() - pos_j + graph.offset_distances[idx_kj]
        else:
            pos_ji = pos[idx_j].detach() - pos_i
            pos_kj = pos[idx_k].detach() - pos_j

        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(graph.edge_distance)
        sbf = self.sbf(graph.edge_distance, angle, idx_kj)

        # ── 初始边嵌入 ────────────────────────────────────────────────────────
        x = self.emb(z, rbf, i, j)   # [E, hidden_channels]

        # ── nm FiLM 注入（边级，Layer 0）────────────────────────────────────
        # _get_nm_per_node 返回 [N, nm_encoded_dim]，用 i（目标原子）索引到边
        # batch[i] == batch[j]（同一结构内的边），用哪个都一样
        nm_per_node = self._get_nm_per_node(data, batch, num_atoms, device)  # [N, D]
        nm_per_edge = nm_per_node[i]                                          # [E, D]
        gamma_nm = self.nm_gamma_mlp(nm_per_edge)   # [E, hidden_channels]
        beta_nm  = self.nm_beta_mlp(nm_per_edge)    # [E, hidden_channels]
        x = x * (1.0 + gamma_nm) + beta_nm

        # ── atom FiLM 注入（边级，Layer 0）──────────────────────────────────
        # 边 (i→j) 连接原子对，对 i 和 j 各算一次 gamma/beta 再相加
        # zero-init → 初始 gamma_i=0, gamma_j=0, 总 gamma=0，训练稳定
        if self.atom_encoder is not None and self.atom_gamma_mlp is not None:
            atom_feat = self.atom_encoder(z)                      # [N, atom_extra_dim]
            gamma_atom = (self.atom_gamma_mlp(atom_feat[i]) +
                          self.atom_gamma_mlp(atom_feat[j]))      # [E, hidden_channels]
            beta_atom  = (self.atom_beta_mlp(atom_feat[i]) +
                          self.atom_beta_mlp(atom_feat[j]))       # [E, hidden_channels]
            x = x * (1.0 + gamma_atom) + beta_atom

        # ── global FiLM 注入（边级，Layer 0）────────────────────────────────
        # 与 nm 同层注入，全局信息从初始边嵌入就开始影响所有 4 个 block
        # zero-init → 初始 gamma=0, beta=0，不干扰 DimeNet++ 起点
        if self.mof_global_encoder is not None and self.global_gamma_mlp is not None:
            global_per_node = self._get_global_per_node(data, batch, device)  # [N, G] or None
            if global_per_node is not None:
                global_per_edge = global_per_node[i]                           # [E, G]
                gamma_g = self.global_gamma_mlp(global_per_edge)               # [E, hidden_channels]
                beta_g  = self.global_beta_mlp(global_per_edge)               # [E, hidden_channels]
                x = x * (1.0 + gamma_g) + beta_g

        # ── DimeNet++ 多尺度前向 ──────────────────────────────────────────────
        # output_block.lin 零初始化 → P=0 at init → MAE 梯度清晰
        P = self.output_blocks[0](x, rbf, i, num_nodes=num_atoms)  # [N, 1]

        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i, num_nodes=num_atoms)      # [N, 1]

        return {"node_features": P, "batch": batch}

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Weighted Energy Head
# ---------------------------------------------------------------------------

@registry.register_model("dimenetplusplus_global_weighted_energy_head")
class DimeNetPlusPlusGlobalWeightedEnergyHead(nn.Module, HeadInterface):
    """PhAST-style weighted energy head for DimeNetPlusPlusGlobal.

    E_total = Σ_i  w_i · e_i
    where e_i = energy_mlp(h_i),  w_i = sigmoid(weight_mlp(h_i)).

    Mirrors SchNetGlobalWeightedEnergyHead — operates on x [N, hidden_channels].
    """

    def __init__(
        self,
        backbone,
        reduce: str = "weighted_sum",
        weight_nn_hidden_dim: int = 64,
        use_initial_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__()

        if isinstance(backbone, dict):
            bb_cfg = dict(backbone)
            bb_name = bb_cfg.pop("name", "dimenetplusplus_global")
            backbone = registry.get_model_class(bb_name)(**bb_cfg)

        self.backbone = backbone
        self.reduce = reduce
        self.use_initial_embeddings = use_initial_embeddings

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

        # ── 全局加性能量修正 ────────────────────────────────────────────────
        # E_total = E_local (DimeNet++) + E_global (per-system offset)
        # 设计原则：全局特征描述体系偏置，不干预局部边特征的消息传递
        # zero-init 保证初始 offset=0，不破坏 DimeNet++ 的训练起点
        self.global_offset_mlp = None
        if (hasattr(backbone, 'mof_global_encoder') and
                backbone.mof_global_encoder is not None and
                hasattr(backbone, 'mof_global_dim')):
            G = backbone.mof_global_dim
            self.global_offset_mlp = nn.Sequential(
                nn.Linear(G, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(self.global_offset_mlp[-1].weight)
            nn.init.zeros_(self.global_offset_mlp[-1].bias)

    def forward(
        self,
        data,
        emb: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if emb is None:
            emb = self.backbone.forward_embeddings(data)

        x = emb["node_features"]   # [N, 1] from output_blocks (DimeNet++ original path)
                                   # or [N, hidden_channels] if node-feature path is used

        num_systems = len(data.natoms)
        batch = emb.get("batch", data.batch)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # DimeNet++ original path: output_blocks already produce [N, 1] per-atom energy.
        # Skip energy_mlp / weight_mlp — directly scatter-sum to system energy.
        if x.size(-1) == 1:
            energy = scatter(
                x.view(-1), batch,
                dim=0, dim_size=num_systems, reduce='add'
            )
            # 加性全局偏置：E_total = E_local + global_offset_mlp(global_emb)
            if self.global_offset_mlp is not None and "global_emb" in emb:
                energy = energy + self.global_offset_mlp(emb["global_emb"]).view(-1)
            return {"energy": energy}

        # Node-feature path (future: nm/global injection returns [N, hidden_channels])
        node_energy = self.energy_mlp(x)   # [N, 1]

        if self.weight_mlp is not None:
            node_weights = self.weight_mlp(x)
            if self.weight_mlp_init is not None:
                node_weights = node_weights + self.weight_mlp_init(x)
            weighted_node_energy = node_energy * node_weights
        else:
            weighted_node_energy = node_energy
            node_weights = torch.ones_like(node_energy)

        energy = scatter(
            weighted_node_energy.view(-1), batch,
            dim=0, dim_size=num_systems, reduce='add'
        )

        if self.reduce in ("weighted_sum_normalized", "mean"):
            weight_sums = scatter(
                node_weights.view(-1), batch,
                dim=0, dim_size=num_systems, reduce='add'
            )
            weight_sums = torch.clamp(weight_sums, min=1e-6)
            energy = energy / weight_sums

        return {"energy": energy}
