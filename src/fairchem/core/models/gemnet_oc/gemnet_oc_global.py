"""
GemNet-OC + MOF Global Feature Injection
MOF backbone comparison experiment (PhAST backbone suite).

Backbone: GemNet-OC (4-hop, quadruplet-capable, scalar node features)
Injection 设计（节点特征 h [N, emb_size_atom] — 天然适合全局注入）:
  Layer 0   : nm FiLM 条件注入（初始原子嵌入之后）
              h = h * (1 + gamma_nm) + beta_nm
  Block K   : global FiLM 注入（MOFGlobalEncoderV2）
              h = h * (1 + gamma_g) + beta_g

与 DimeNet++ 的关键区别：
  - GemNet-OC 有真正的节点特征 h [N, emb_size_atom]
  - 全局注入直接作用于 h，无需通过边特征间接影响
  - 这使得全局 MOF 拓扑/孔道信息可以直接影响原子表示
  - 注入方式对 scalar backbone 更自然、更完整

能量预测：PhAST 风格加权 energy head（WeightedEnergyHead）
所有注入模块均 zero-init → 初始等价于无注入，训练稳定。
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

from .gemnet_oc import GemNetOCBackbone
from .layers.base_layers import Dense, ResidualLayer


# ---------------------------------------------------------------------------
# Backbone: GemNetOC + nm/global injection on TRUE node features h
# ---------------------------------------------------------------------------

@registry.register_model("gemnet_oc_global")
class GemNetOCGlobal(GemNetOCBackbone):
    """GemNet-OC backbone with MOF condition + global feature injection.

    The key architectural advantage over DimeNet++:
      GemNet-OC has TRUE node features h [N, emb_size_atom].
      Global FiLM operates directly on h — a natural fit for system-level
      MOF properties (topology, pore descriptors).

    Injection strategy (all zero-init → identity at init):
      - Layer 0  : nm FiLM — h = h*(1+γ_nm) + β_nm  (before edge_emb)
      - Block K  : global FiLM — h = h*(1+γ_g) + β_g  at global_inject_layer

    Two-stage training:
      Stage 1: freeze_first_n_layers=0  → train everything
      Stage 2: freeze_first_n_layers=N  → freeze emb + int_blocks[0:N],
               train global FiLM + int_blocks[N:] + energy head.
    """

    def __init__(
        self,
        # --- Standard GemNet-OC params (required) ---
        num_spherical: int = 7,
        num_radial: int = 128,
        num_blocks: int = 4,
        emb_size_atom: int = 256,
        emb_size_edge: int = 512,
        emb_size_trip_in: int = 64,
        emb_size_trip_out: int = 64,
        emb_size_quad_in: int = 32,
        emb_size_quad_out: int = 32,
        emb_size_aint_in: int = 64,
        emb_size_aint_out: int = 64,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_sbf: int = 32,
        num_before_skip: int = 2,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        num_output_afteratom: int = 3,
        num_atom_emb_layers: int = 2,
        num_global_out_layers: int = 2,
        regress_forces: bool = False,
        direct_forces: bool = False,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        scale_backprop_forces: bool = False,
        cutoff: float = 6.0,
        cutoff_qint: float | None = None,
        cutoff_aeaint: float | None = None,
        cutoff_aint: float | None = None,
        max_neighbors: int = 50,
        max_neighbors_qint: int | None = None,
        max_neighbors_aeaint: int | None = None,
        max_neighbors_aint: int | None = None,
        enforce_max_neighbors_strictly: bool = True,
        rbf: dict | None = None,
        rbf_spherical: dict | None = None,
        envelope: dict | None = None,
        cbf: dict | None = None,
        sbf: dict | None = None,
        extensive: bool = True,
        forces_coupled: bool = False,
        output_init: str = "HeOrthogonal",
        activation: str = "silu",
        quad_interaction: bool = False,
        atom_edge_interaction: bool = False,
        edge_atom_interaction: bool = False,
        atom_interaction: bool = False,
        scale_basis: bool = False,
        qint_tags: list | None = None,
        num_elements: int = 83,
        otf_graph: bool = False,
        scale_file: str | None = None,
        # --- Condition (nm) encoder ---
        base_nm_dim: int = 2,
        condition_hidden_dim: int = 64,
        nm_max_count: float = 15.0,
        # --- Atom extra features (AtomPropertyEncoderV2) ---
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
        adapter_hidden_dim: int = 256,
        adapter_dropout: float = 0.0,
        adapter_zero_init: bool = True,
        # --- Injection position ---
        global_inject_layer: int = 2,
        # --- Normalization on global encoder output before FiLM ---
        global_layer_norm: bool = False,
        # --- Two-stage freeze ---
        freeze_first_n_layers: int = 0,
    ) -> None:
        super().__init__(
            num_spherical=num_spherical,
            num_radial=num_radial,
            num_blocks=num_blocks,
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_trip_in=emb_size_trip_in,
            emb_size_trip_out=emb_size_trip_out,
            emb_size_quad_in=emb_size_quad_in,
            emb_size_quad_out=emb_size_quad_out,
            emb_size_aint_in=emb_size_aint_in,
            emb_size_aint_out=emb_size_aint_out,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            emb_size_sbf=emb_size_sbf,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_concat=num_concat,
            num_atom=num_atom,
            num_output_afteratom=num_output_afteratom,
            num_atom_emb_layers=num_atom_emb_layers,
            num_global_out_layers=num_global_out_layers,
            regress_forces=regress_forces,
            direct_forces=direct_forces,
            use_pbc=use_pbc,
            use_pbc_single=use_pbc_single,
            scale_backprop_forces=scale_backprop_forces,
            cutoff=cutoff,
            cutoff_qint=cutoff_qint,
            cutoff_aeaint=cutoff_aeaint,
            cutoff_aint=cutoff_aint,
            max_neighbors=max_neighbors,
            max_neighbors_qint=max_neighbors_qint,
            max_neighbors_aeaint=max_neighbors_aeaint,
            max_neighbors_aint=max_neighbors_aint,
            enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
            rbf=rbf,
            rbf_spherical=rbf_spherical,
            envelope=envelope,
            cbf=cbf,
            sbf=sbf,
            extensive=extensive,
            forces_coupled=forces_coupled,
            output_init=output_init,
            activation=activation,
            quad_interaction=quad_interaction,
            atom_edge_interaction=atom_edge_interaction,
            edge_atom_interaction=edge_atom_interaction,
            atom_interaction=atom_interaction,
            scale_basis=scale_basis,
            qint_tags=qint_tags,
            num_elements=num_elements,
            otf_graph=otf_graph,
            scale_file=scale_file,
        )

        # Save config
        self.emb_size_atom = emb_size_atom
        self.base_nm_dim = base_nm_dim
        self.use_mof_global_features = bool(use_mof_global_features)
        self.mof_global_dim = int(mof_global_dim)
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.global_inject_layer = int(global_inject_layer)
        self.global_layer_norm = bool(global_layer_norm)
        self.freeze_first_n_layers = int(freeze_first_n_layers)
        self.use_atom_extra_features = bool(use_atom_extra_features)
        self.atom_extra_dim = int(atom_extra_dim)

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

        # ── AtomPropertyEncoderV2 (原子物理特征 → atom_extra_dim) ────────────
        self.atom_encoder = None
        if self.use_atom_extra_features:
            try:
                if atom_encoder_type.lower() == "v2":
                    from fairchem.core.models.equiformer_v2.atomic_emb_v2 import AtomPropertyEncoderV2
                    self.atom_encoder = AtomPropertyEncoderV2(
                        max_Z=num_elements, out_dim=self.atom_extra_dim
                    )
                else:
                    from fairchem.core.models.equiformer_v2.atomic_emb import AtomPropertyEncoder
                    self.atom_encoder = AtomPropertyEncoder(
                        max_Z=num_elements, out_dim=self.atom_extra_dim
                    )
                logging.info(f"[GemNetOCGlobal] AtomPropertyEncoder ({atom_encoder_type}) init OK, out_dim={self.atom_extra_dim}")
            except Exception as e:
                logging.warning(f"[GemNetOCGlobal] AtomPropertyEncoder init failed: {e}")
                self.atom_encoder = None

        # context = nm_encoded [64] + atom_feat [128] (if enabled)
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
                logging.info("MOFGlobalEncoderV2 (三模态) initialised for GemNet-OC")
            except Exception as e:
                logging.warning(f"MOFGlobalEncoderV2 init failed: {e}")

        def _zero_last(seq: nn.Sequential) -> None:
            for m in reversed(list(seq)):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    return

        # ── context FiLM (Layer 0, operates on h [N, emb_size_atom]) ─────────
        # context = nm_encoded [N, nm_encoded_dim]
        #         + atom_feat  [N, atom_extra_dim]  (if use_atom_extra_features)
        # input dim = context_input_dim (64 or 192)
        #
        # 保护 1：context 整体先归一化（对齐 eSCN 的 context_ln）
        self.context_ln = nn.LayerNorm(self.context_input_dim)
        #
        # 保护 2：hidden 层 LayerNorm（对齐 eSCN 的 film_gamma 结构）
        self.nm_gamma_mlp = nn.Sequential(
            nn.Linear(self.context_input_dim, adapter_hidden_dim),
            nn.LayerNorm(adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, emb_size_atom),
        )
        self.nm_beta_mlp = nn.Sequential(
            nn.Linear(self.context_input_dim, adapter_hidden_dim),
            nn.LayerNorm(adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, emb_size_atom),
        )
        if adapter_zero_init:
            _zero_last(self.nm_gamma_mlp)
            _zero_last(self.nm_beta_mlp)

        # ── global FiLM (Block K, operates on h [N, emb_size_atom]) ─────────
        self.global_gamma_mlp: nn.Sequential | None = None
        self.global_beta_mlp: nn.Sequential | None = None
        self.global_input_norm: nn.LayerNorm | None = None
        if self.use_mof_global_features:
            # optional LayerNorm on encoder output (decouples pretrained scale)
            if self.global_layer_norm:
                self.global_input_norm = nn.LayerNorm(mof_global_dim)
            self.global_gamma_mlp = nn.Sequential(
                nn.Linear(mof_global_dim, adapter_hidden_dim),
                nn.LayerNorm(adapter_hidden_dim) if self.global_layer_norm else nn.Identity(),
                nn.GELU(),
                nn.Linear(adapter_hidden_dim, emb_size_atom),
            )
            self.global_beta_mlp = nn.Sequential(
                nn.Linear(mof_global_dim, adapter_hidden_dim),
                nn.LayerNorm(adapter_hidden_dim) if self.global_layer_norm else nn.Identity(),
                nn.GELU(),
                nn.Linear(adapter_hidden_dim, emb_size_atom),
            )
            if adapter_zero_init:
                _zero_last(self.global_gamma_mlp)
                _zero_last(self.global_beta_mlp)

        # ── Two-stage freeze ─────────────────────────────────────────────────
        if freeze_first_n_layers > 0:
            # basis functions
            self.radial_basis.requires_grad_(False)
            self.cbf_basis_tint.requires_grad_(False)
            self.mlp_rbf_tint.requires_grad_(False)
            self.mlp_cbf_tint.requires_grad_(False)
            self.mlp_rbf_h.requires_grad_(False)
            self.mlp_rbf_out.requires_grad_(False)
            # atom + edge embeddings
            self.atom_emb.requires_grad_(False)
            self.edge_emb.requires_grad_(False)
            # interaction blocks
            n = min(freeze_first_n_layers, len(self.int_blocks))
            for i in range(n):
                self.int_blocks[i].requires_grad_(False)
            # output blocks (freeze corresponding out_blocks)
            for i in range(n + 1):
                self.out_blocks[i].requires_grad_(False)
            # nm encoder + context FiLM (including atom encoder if present)
            self.condition_encoder.requires_grad_(False)
            self.nm_gamma_mlp.requires_grad_(False)
            self.nm_beta_mlp.requires_grad_(False)
            if self.atom_encoder is not None:
                self.atom_encoder.requires_grad_(False)
            logging.info(
                f"[GemNetOCGlobal] Frozen: radial_basis + cbf + basis_mlps + "
                f"atom_emb + edge_emb + int_blocks[0:{n}] + out_blocks[0:{n+1}] + "
                f"condition_encoder + atom_encoder + context_film"
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
            return nm_encoded[batch]  # [N, nm_encoded_dim]
        else:
            return torch.zeros(
                num_atoms, self.nm_encoded_dim, device=device, dtype=torch.float32
            )

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

    # ── Main forward (overrides GemNetOCBackbone.forward) ───────────────────

    @conditional_grad(torch.enable_grad())
    def forward(self, data) -> dict[str, torch.Tensor]:
        """GemNet-OC forward with nm + global FiLM injection on node features h.

        Key: h [N, emb_size_atom] is a TRUE per-atom feature.
        nm injection at Layer 0 (after atom_emb, before edge_emb).
        global injection at Block K (after int_blocks[K-1]).

        数据流：
          atom_emb(z) → h [N, emb_size_atom]
          nm FiLM: h = h * (1 + γ_nm) + β_nm     ← nm 注入点（节点级，Layer 0）
          edge_emb(h, rbf) → m [E, emb_size_edge]
          out_blocks[0](h, m) → x_E [N, emb_size_atom]
          for i in range(num_blocks):
              h, m = int_blocks[i](h, m, ...)      ← 消息传递
              if i == global_inject_layer - 1:
                  h = h * (1 + γ_g) + β_g         ← global 注入点（节点级）
              x_E = out_blocks[i+1](h, m)          → [N, emb_size_atom]
              xs_E.append(x_E)
          return xs_E for weighted energy head
        """
        pos = data.pos
        batch = data.batch
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = atomic_numbers.shape[0]
        device = pos.device

        (
            main_graph,
            a2a_graph,
            a2ee2a_graph,
            qint_graph,
            id_swap,
            trip_idx_e2e,
            trip_idx_a2e,
            trip_idx_e2a,
            quad_idx,
        ) = self.get_graphs_and_indices(data)
        _, idx_t = main_graph["edge_index"]

        (
            basis_rad_raw,
            basis_atom_update,
            basis_output,
            bases_qint,
            bases_e2e,
            bases_a2e,
            bases_e2a,
            basis_a2a_rad,
        ) = self.get_bases(
            main_graph=main_graph,
            a2a_graph=a2a_graph,
            a2ee2a_graph=a2ee2a_graph,
            qint_graph=qint_graph,
            trip_idx_e2e=trip_idx_e2e,
            trip_idx_a2e=trip_idx_a2e,
            trip_idx_e2a=trip_idx_e2a,
            quad_idx=quad_idx,
            num_atoms=num_atoms,
        )

        # ── Atom embedding ────────────────────────────────────────────────────
        h = self.atom_emb(atomic_numbers)
        # (nAtoms, emb_size_atom)

        # ── Context FiLM (Layer 0, node-level) ────────────────────────────────
        # context = nm_encoded [N, 64] + atom_feat [N, 128] (if enabled)
        nm_per_node = self._get_nm_per_node(data, batch, num_atoms, device)  # [N, nm_encoded_dim]
        context_parts = [nm_per_node]
        if self.use_atom_extra_features and self.atom_encoder is not None:
            tags = data.tags if hasattr(data, "tags") else None
            atom_feat = self.atom_encoder(atomic_numbers, tags=tags).to(device=device)  # [N, atom_extra_dim]
            context_parts.append(atom_feat)
        context = torch.cat(context_parts, dim=-1)  # [N, context_input_dim]
        context = self.context_ln(context)           # 保护 1：归一化 context
        gamma_nm = self.nm_gamma_mlp(context)   # [N, emb_size_atom]
        beta_nm  = self.nm_beta_mlp(context)    # [N, emb_size_atom]
        h = h * (1.0 + gamma_nm) + beta_nm

        # ── Edge embedding (uses nm-injected h) ───────────────────────────────
        m = self.edge_emb(h, basis_rad_raw, main_graph["edge_index"])
        # (nEdges, emb_size_edge)

        # ── Pre-interaction output block ──────────────────────────────────────
        x_E, x_F = self.out_blocks[0](h, m, basis_output, idx_t)
        # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
        xs_E, xs_F = [x_E], [x_F]

        # ── Global embedding lookup (once, reused at inject layer) ────────────
        global_per_node = None
        if self.mof_global_encoder is not None and self.global_gamma_mlp is not None:
            global_per_node = self._get_global_per_node(data, batch, device)
            if global_per_node is not None and self.global_input_norm is not None:
                global_per_node = self.global_input_norm(global_per_node)

        # ── global FiLM at Layer 0 (before any blocks, same level as nm) ──────
        # global_inject_layer=0: inject here where h magnitude ≈ 1-2 (stable)
        if self.global_inject_layer == 0 and global_per_node is not None:
            gamma_g = self.global_gamma_mlp(global_per_node)  # [N, emb_size_atom]
            beta_g  = self.global_beta_mlp(global_per_node)   # [N, emb_size_atom]
            h = h * (1.0 + gamma_g) + beta_g

        # ── Interaction blocks with global FiLM injection ─────────────────────
        for i in range(self.num_blocks):
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                bases_qint=bases_qint,
                bases_e2e=bases_e2e,
                bases_a2e=bases_a2e,
                bases_e2a=bases_e2a,
                basis_a2a_rad=basis_a2a_rad,
                basis_atom_update=basis_atom_update,
                edge_index_main=main_graph["edge_index"],
                a2ee2a_graph=a2ee2a_graph,
                a2a_graph=a2a_graph,
                id_swap=id_swap,
                trip_idx_e2e=trip_idx_e2e,
                trip_idx_a2e=trip_idx_a2e,
                trip_idx_e2a=trip_idx_e2a,
                quad_idx=quad_idx,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            # ── global FiLM at block K (global_inject_layer >= 1) ─────────────
            # 在局部结构捕获（前K块）之后注入全局信息
            if (self.global_inject_layer >= 1
                    and i == self.global_inject_layer - 1
                    and global_per_node is not None):
                gamma_g = self.global_gamma_mlp(global_per_node)  # [N, emb_size_atom]
                beta_g  = self.global_beta_mlp(global_per_node)   # [N, emb_size_atom]
                h = h * (1.0 + gamma_g) + beta_g

            x_E, x_F = self.out_blocks[i + 1](h, m, basis_output, idx_t)
            # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
            xs_E.append(x_E)
            xs_F.append(x_F)

        return {
            "xs_E": xs_E,
            "xs_F": xs_F,
            "edge_vec": main_graph["vector"],
            "edge_idx": idx_t,
            "num_neighbors": main_graph["num_neighbors"],
            "batch": batch,
        }

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Weighted Energy Head for GemNet-OC Global
# ---------------------------------------------------------------------------

@registry.register_model("gemnet_oc_global_weighted_energy_head")
class GemNetOCGlobalWeightedEnergyHead(nn.Module, HeadInterface):
    """PhAST-style weighted energy head for GemNetOCGlobal.

    Processes xs_E (list of per-atom embeddings from each block) via:
      1. out_mlp_E(cat(xs_E)) → h_final [N, emb_size_atom]
      2. E_i = energy_mlp(h_final_i)    [N, 1]
      3. w_i = weight_mlp(h_final_i)    [N, 1]  (sigmoid gated)
      4. E_total = scatter(E_i * w_i, batch, reduce='add')

    This replaces GemNetOC's standard out_mlp_E + out_energy + scatter.
    """

    def __init__(
        self,
        backbone,
        reduce: str = "weighted_sum",
        weight_nn_hidden_dim: int = 64,
        num_global_out_layers: int = 2,
        **kwargs,
    ):
        super().__init__()

        if isinstance(backbone, dict):
            bb_cfg = dict(backbone)
            bb_name = bb_cfg.pop("name", "gemnet_oc_global")
            backbone = registry.get_model_class(bb_name)(**bb_cfg)

        self.backbone = backbone
        self.reduce = reduce

        C = backbone.atom_emb.emb_size
        num_blocks = len(backbone.int_blocks)

        # Nullify backbone's original output head to avoid redundant params
        backbone.out_mlp_E = None
        backbone.out_energy = None

        # ── Multi-scale aggregation: cat(xs_E) → [N, C*(num_blocks+1)] → [N, C] ──
        out_mlp_E = [
            Dense(
                C * (num_blocks + 1),
                C,
                activation=backbone.activation,
            )
        ] + [
            ResidualLayer(
                C,
                activation=backbone.activation,
            )
            for _ in range(num_global_out_layers)
        ]
        self.out_mlp_E = nn.Sequential(*out_mlp_E)

        # ── Per-atom energy MLP ────────────────────────────────────────────────
        self.energy_mlp = nn.Sequential(
            nn.Linear(C, weight_nn_hidden_dim),
            nn.SiLU(),
            nn.Linear(weight_nn_hidden_dim, weight_nn_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(weight_nn_hidden_dim // 2, 1),
        )

        # ── Per-atom weight MLP ────────────────────────────────────────────────
        if "weighted" in reduce:
            self.weight_mlp = nn.Sequential(
                nn.Linear(C, weight_nn_hidden_dim),
                nn.SiLU(),
                nn.Linear(weight_nn_hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.weight_mlp = None

    def forward(
        self,
        data,
        emb: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if emb is None:
            emb = self.backbone(data)

        xs_E = emb["xs_E"]  # list of [N, emb_size_atom], length = num_blocks + 1
        batch = emb.get("batch", data.batch)

        # ── Multi-scale feature aggregation ─────────────────────────────────
        h = self.out_mlp_E(torch.cat(xs_E, dim=-1))  # [N, emb_size_atom]

        # ── Per-atom energy and weight ────────────────────────────────────────
        node_energy = self.energy_mlp(h)              # [N, 1]

        if self.weight_mlp is not None:
            node_weights = self.weight_mlp(h)         # [N, 1]
            weighted_node_energy = node_energy * node_weights
        else:
            weighted_node_energy = node_energy
            node_weights = torch.ones_like(node_energy)

        num_systems = len(data.natoms)
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
