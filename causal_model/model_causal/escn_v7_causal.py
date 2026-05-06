"""
V7 (CMG) eSCN-based causal-embedded model classes.

Three stages, all from-scratch trained (no warmstart from existing checkpoints):

  EscnV7CausalS1 — atom-level causal foundation
                   = eSCN base + atom_extra_features V2 + atom_type head

  EscnV7CausalS2 — motif-level causal augmentation
                   = (frozen S1) + mof_global_features (smi-ted/selfies-ted/MHG)
                     + IRM invariance head

  EscnV7CausalS3 — mechanism-level causal embedding
                   = (frozen S2) + AtomMaskHead + DualEnergyHead + CooperativeHead

Each is a wrapper around the existing eSCN model. Source eSCN code is NOT
modified. We register new model names so existing yml's still work.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fairchem.core.common.registry import registry
from .causal_modules import (
    AtomMaskHead, AtomTypeHead, DualEnergyHead, CooperativeHead,
    neighbor_summary_simple,
)


def _get_existing_model_class(name: str):
    cls = registry.get_model_class(name)
    if cls is None:
        raise RuntimeError(f"Required existing model '{name}' not registered. "
                           "Did you forget to import irm_trainer first?")
    return cls


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — atom-level causal foundation
# ─────────────────────────────────────────────────────────────────────
@registry.register_model("escn_v7_causal_s1")
class EscnV7CausalS1(nn.Module):
    """V7 stage 1: eSCN + atom_extra_features V2 + atom_type head.

    Acts as a thin wrapper around the existing escn_weighted_energy_head model.
    Adds an atom-type prediction head fed from the l=0 invariant features.

    The forward returns:
        outputs["energy"]            : same as eSCN base
        outputs["atom_type_logits"]  : (N, n_classes) for L_atom_type
    """

    def __init__(
        self,
        backbone: dict | None = None,
        # The following keys mirror eSCNWeightedEnergyHead defaults
        reduce: str = "weighted_sum",
        weight_nn_hidden_dim: int = 64,
        use_initial_embeddings: bool = True,
        condition_dim: int = 2,
        condition_hidden_dim: int = 64,
        # New for V7-S1
        atom_type_n_classes: int = 100,
        atom_type_hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        # Build the existing model via registry (no source modification)
        base_cls = _get_existing_model_class("escn_weighted_energy_head")
        self.base = base_cls(
            backbone=backbone,
            reduce=reduce,
            weight_nn_hidden_dim=weight_nn_hidden_dim,
            use_initial_embeddings=use_initial_embeddings,
            condition_dim=condition_dim,
            condition_hidden_dim=condition_hidden_dim,
            **kwargs,
        )

        sphere_channels = int(self.base.backbone.sphere_channels)
        self.atom_type_head = AtomTypeHead(
            in_dim=sphere_channels, n_classes=atom_type_n_classes
        )
        self.atom_type_n_classes = atom_type_n_classes

    @property
    def backbone(self):
        return self.base.backbone

    def forward(self, data) -> dict:
        emb = self.backbone.forward_embeddings(data)
        sphere_values = emb["sphere_values"]   # (N*samples, channels)
        ns = int(self.backbone.num_sphere_samples)
        sc = int(self.backbone.sphere_channels)
        per_atom = sphere_values.view(-1, ns, sc).mean(dim=1)   # (N, sphere_channels)

        # IMPORTANT: detach per_atom before atom_type_head so the auxiliary
        # CE loss does NOT back-propagate into the backbone. This avoids
        # gradient explosion caused by conflicting energy/atom_type signals
        # competing on the same shared sphere_values activations.
        atom_type_logits = self.atom_type_head(per_atom.detach())  # (N, n_classes)

        # Run original head to get energy (gradient still flows here normally)
        out = self.base(data, emb=emb)
        out["atom_type_logits"] = atom_type_logits
        out["per_atom_features"] = per_atom        # not detached, used elsewhere
        return out


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — motif-level (frozen S1 + global features + IRM)
# ─────────────────────────────────────────────────────────────────────
@registry.register_model("escn_v7_causal_s2")
class EscnV7CausalS2(nn.Module):
    """V7 stage 2: built on frozen S1, adds global features + IRM.

    Implementation: builds an escn_weighted_energy_head with V2 features turned
    on. The S1 weights are loaded externally (via load_state_dict from S1
    checkpoint) and frozen for the part that overlaps. Atom_type head is
    inherited from S1.
    """

    def __init__(
        self,
        backbone: dict | None = None,
        reduce: str = "weighted_sum",
        weight_nn_hidden_dim: int = 64,
        use_initial_embeddings: bool = True,
        condition_dim: int = 2,
        condition_hidden_dim: int = 64,
        atom_type_n_classes: int = 100,
        atom_type_hidden_dim: int = 64,
        # Stage 2 specific: which blocks to freeze (default first 6)
        freeze_first_n_layers: int = 6,
        **kwargs,
    ):
        super().__init__()
        base_cls = _get_existing_model_class("escn_weighted_energy_head")
        self.base = base_cls(
            backbone=backbone,
            reduce=reduce,
            weight_nn_hidden_dim=weight_nn_hidden_dim,
            use_initial_embeddings=use_initial_embeddings,
            condition_dim=condition_dim,
            condition_hidden_dim=condition_hidden_dim,
            **kwargs,
        )
        sphere_channels = int(self.base.backbone.sphere_channels)
        self.atom_type_head = AtomTypeHead(
            in_dim=sphere_channels, n_classes=atom_type_n_classes
        )
        self.freeze_first_n_layers = freeze_first_n_layers

    @property
    def backbone(self):
        return self.base.backbone

    def freeze_s1_weights(self):
        """Freeze backbone first N layers + sphere_embedding + distance_expansion.
        Called by trainer after loading S1 weights."""
        bb = self.backbone
        if hasattr(bb, "sphere_embedding"):
            for p in bb.sphere_embedding.parameters():
                p.requires_grad = False
        if hasattr(bb, "distance_expansion"):
            for p in bb.distance_expansion.parameters():
                p.requires_grad = False
        if hasattr(bb, "layer_blocks"):
            n = min(self.freeze_first_n_layers, len(bb.layer_blocks))
            for i in range(n):
                for p in bb.layer_blocks[i].parameters():
                    p.requires_grad = False
        # Atom_extra_features encoder also frozen (S1 component)
        if hasattr(bb, "atom_encoder") and bb.atom_encoder is not None:
            for p in bb.atom_encoder.parameters():
                p.requires_grad = False
        for p in self.atom_type_head.parameters():
            p.requires_grad = False

    def forward(self, data) -> dict:
        emb = self.backbone.forward_embeddings(data)
        sphere_values = emb["sphere_values"]
        ns = int(self.backbone.num_sphere_samples)
        sc = int(self.backbone.sphere_channels)
        per_atom = sphere_values.view(-1, ns, sc).mean(dim=1)
        atom_type_logits = self.atom_type_head(per_atom)

        out = self.base(data, emb=emb)
        out["atom_type_logits"] = atom_type_logits
        out["per_atom_features"] = per_atom
        return out


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — mechanism-level (frozen S2 + mask + dual head + cooperative)
# ─────────────────────────────────────────────────────────────────────
@registry.register_model("escn_v7_causal_s3")
class EscnV7CausalS3(nn.Module):
    """V7 stage 3: AtomMaskHead + DualEnergyHead + CooperativeHead.

    Stage 2 backbone is frozen entirely. Only the new causal modules train.
    Forward returns:
        outputs["energy"]               : E_C (causal branch)  ← used for prediction
        outputs["energy_context"]       : E_S (context branch) ← used in L_S loss
        outputs["alpha"]                : (N, C) per-atom × per-channel mask
        outputs["cooperative_pred"]     : (B,) per-graph cooperative term
        outputs["per_atom_features"]    : (N, C) raw l=0 feats (for analysis)
    """

    def __init__(
        self,
        backbone: dict | None = None,
        reduce: str = "weighted_sum",
        weight_nn_hidden_dim: int = 64,
        use_initial_embeddings: bool = True,
        condition_dim: int = 2,
        condition_hidden_dim: int = 64,
        # Causal modules
        mask_hidden_dim: int = 64,
        coop_hidden_dim: int = 32,
        coop_max_count: float = 5.0,
        **kwargs,
    ):
        super().__init__()
        base_cls = _get_existing_model_class("escn_weighted_energy_head")
        self.base = base_cls(
            backbone=backbone,
            reduce=reduce,
            weight_nn_hidden_dim=weight_nn_hidden_dim,
            use_initial_embeddings=use_initial_embeddings,
            condition_dim=condition_dim,
            condition_hidden_dim=condition_hidden_dim,
            **kwargs,
        )
        sphere_channels = int(self.base.backbone.sphere_channels)

        self.mask_head = AtomMaskHead(
            in_dim=sphere_channels, hidden_dim=mask_hidden_dim,
            use_prior=True, n_neighbor_types=5,
        )
        self.dual_head = DualEnergyHead(in_dim=sphere_channels, hidden_dim=mask_hidden_dim)
        self.coop_head = CooperativeHead(
            ctx_dim=sphere_channels, hidden_dim=coop_hidden_dim,
            max_count=coop_max_count,
        )

    @property
    def backbone(self):
        return self.base.backbone

    def freeze_s2_weights(self):
        """Freeze the entire base model. Only causal modules train."""
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, data) -> dict:
        # Run base eSCN (frozen) to get sphere_values
        emb = self.backbone.forward_embeddings(data)
        sphere_values = emb["sphere_values"]
        ns = int(self.backbone.num_sphere_samples)
        sc = int(self.backbone.sphere_channels)
        per_atom = sphere_values.view(-1, ns, sc).mean(dim=1)   # (N, C)

        # Build neighbor summary (uses graph from emb["graph"])
        graph = emb["graph"]
        edge_index = graph.edge_index
        neighbor_summary = neighbor_summary_simple(
            data.atomic_numbers.long(), edge_index, n_categories=5,
        ).to(per_atom.dtype)

        # Get prior_α: pulled from data.prior_alpha if attached, else None
        prior_alpha = getattr(data, "prior_alpha_per_atom", None)
        if prior_alpha is not None:
            prior_alpha = prior_alpha.to(per_atom.dtype)

        # Compute mask α
        alpha = self.mask_head(per_atom, prior_alpha, neighbor_summary)   # (N, C)

        # Causal / context split
        h_C = per_atom * alpha
        h_S = per_atom * (1.0 - alpha)

        # Dual energy heads
        n_graphs = int(data.natoms.shape[0]) if data.natoms.dim() > 0 else 1
        batch = data.batch
        E_C, E_S, node_E_C, node_E_S = self.dual_head(h_C, h_S, batch, n_graphs)

        # Cooperative head
        condition = data.condition.float()   # (B, 2)
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        n_h2o = condition[:, 0]
        n_co2 = condition[:, 1]
        # Per-graph context: mean-pool of h_C
        ctx = torch.zeros(n_graphs, sc, device=per_atom.device, dtype=per_atom.dtype)
        ctx.index_add_(0, batch, h_C)
        atom_counts = torch.zeros(n_graphs, device=per_atom.device, dtype=per_atom.dtype)
        atom_counts.index_add_(0, batch, torch.ones_like(per_atom[:, 0]))
        ctx = ctx / atom_counts.clamp(min=1).unsqueeze(-1)
        coop = self.coop_head(n_h2o, n_co2, ctx)                      # (B,)

        # Final energy = E_C + cooperative (the cooperative term is added on top)
        energy = E_C + coop

        return {
            "energy": energy,
            "energy_context": E_S,
            "alpha": alpha,
            "cooperative_pred": coop,
            "per_atom_features": per_atom,
            "node_E_C": node_E_C,
            "node_E_S": node_E_S,
        }
