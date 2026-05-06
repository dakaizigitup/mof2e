"""
V9 eSCN-based causal-embedded model classes.

Implemented:
  EscnV9CausalS1 — atom semantic foundation
  EscnV9CausalS2 — environment-invariant tri-branch encoding
  EscnV9CausalS3 — mechanism decomposition + cooperative component
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fairchem.core.common.registry import registry
from .causal_modules import AtomTypeHead, CooperativeHead


def _get_existing_model_class(name: str):
    cls = registry.get_model_class(name)
    if cls is None:
        raise RuntimeError(
            f"Required existing model '{name}' not registered. "
            "Did you forget to import irm_trainer first?"
        )
    return cls


def _pool_mean(node_feat: torch.Tensor, batch: torch.Tensor, n_graphs: int):
    out = torch.zeros(
        n_graphs,
        node_feat.size(-1),
        device=node_feat.device,
        dtype=node_feat.dtype,
    )
    out.index_add_(0, batch, node_feat)
    cnt = torch.zeros(n_graphs, device=node_feat.device, dtype=node_feat.dtype)
    cnt.index_add_(0, batch, torch.ones_like(node_feat[:, 0]))
    return out / cnt.clamp(min=1.0).unsqueeze(-1)


def _clone_with_masked_atomic_numbers(data, atom_mask: torch.Tensor, mask_token: int):
    masked = data.clone()
    atomic_numbers = data.atomic_numbers.clone()
    atomic_numbers[atom_mask] = int(mask_token)
    masked.atomic_numbers = atomic_numbers
    return masked


class _TriBranchEncoder(nn.Module):
    """Three structural branches (gas-condition agnostic).

    R-3: motif previously absorbed `condition`, which made it a fused
    structure+gas descriptor that short-circuited IRM and made decorrel
    against site/path trivial. After R-3:
      - z_site  = pool over METAL atoms          (Z>=11, excl. P/S/Cl/Br/I)
      - z_motif = pool over NON-METAL atoms      (organic scaffold)
      - z_path  = pool over ALL atoms            (full-graph context)
    All three are pure structural descriptors. Gas condition is fused
    *outside* the encoder (in tri_energy_head and CooperativeHead).
    """

    # P/S/Cl/Br/I — main-group nonmetals with Z>=11 that must NOT be tagged
    # as metal when building z_site (the metal-coordination representation).
    # Same rule as causal_modules.py:neighbor_summary_simple.
    _MAIN_NONMETAL_Z = (15, 16, 17, 35, 53)

    def __init__(self, in_dim: int = 128, branch_dim: int = 128):
        super().__init__()
        self.site_proj = nn.Sequential(
            nn.Linear(in_dim, branch_dim), nn.GELU(), nn.Linear(branch_dim, branch_dim)
        )
        self.motif_proj = nn.Sequential(
            nn.Linear(in_dim, branch_dim), nn.GELU(), nn.Linear(branch_dim, branch_dim)
        )
        self.path_node_proj = nn.Sequential(
            nn.Linear(in_dim, branch_dim), nn.GELU(), nn.Linear(branch_dim, branch_dim)
        )

    def _is_metal_mask(self, atomic_numbers: torch.Tensor, dtype) -> torch.Tensor:
        main_nonmetal = torch.tensor(
            self._MAIN_NONMETAL_Z, device=atomic_numbers.device, dtype=atomic_numbers.dtype
        )
        return (
            (atomic_numbers >= 11) & ~torch.isin(atomic_numbers, main_nonmetal)
        ).to(dtype).unsqueeze(-1)

    def forward(
        self,
        per_atom: torch.Tensor,
        atomic_numbers: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int,
        path_dropout_prob: float = 0.15,
    ):
        is_metal = self._is_metal_mask(atomic_numbers, per_atom.dtype)
        is_nonmetal = 1.0 - is_metal

        site_node = self.site_proj(per_atom) * is_metal
        z_site = _pool_mean(site_node, batch, n_graphs)

        motif_node = self.motif_proj(per_atom) * is_nonmetal
        z_motif = _pool_mean(motif_node, batch, n_graphs)

        path_node = self.path_node_proj(per_atom)
        z_path = _pool_mean(path_node, batch, n_graphs)

        keep = (torch.rand(path_node.size(0), 1, device=path_node.device) >= path_dropout_prob).to(path_node.dtype)
        z_path_aug = _pool_mean(path_node * keep, batch, n_graphs)

        return z_site, z_motif, z_path, z_path_aug


class CausalMixer(nn.Module):
    """Explicit V9 mixer: E = E_site + E_motif + E_path + E_coop.

    Each structural head takes (branch_features, condition) so it can express
    gas-conditional binding at that mechanism (e.g. metal site binds H2O and
    CO2 with different affinity). The cooperative head still carries the
    h*c hard-zero so E_coop = 0 at pure conditions; what `coop` means here is
    "interaction energy beyond independent contributions", which is the right
    chemistry meaning of cooperativity.

    Counterfactual semantics (used by trainer):
      - zeroing z_motif → "what if the scaffold contributed nothing"
      - swapping z_motif → "what if this MOF had a different scaffold"
    Both pass through the SAME mixer, so the head's gas-conditioning is
    preserved across counterfactuals.
    """

    def __init__(
        self,
        branch_dim: int = 128,
        coop_hidden_dim: int = 64,
        coop_max_count: float = 5.0,
        condition_dim: int = 2,
    ):
        super().__init__()
        # R-3 follow-up: each branch head fuses (branch_z, condition).
        self.site_head = nn.Sequential(
            nn.Linear(branch_dim + condition_dim, branch_dim), nn.GELU(), nn.Linear(branch_dim, 1)
        )
        self.motif_head = nn.Sequential(
            nn.Linear(branch_dim + condition_dim, branch_dim), nn.GELU(), nn.Linear(branch_dim, 1)
        )
        self.path_head = nn.Sequential(
            nn.Linear(branch_dim + condition_dim, branch_dim), nn.GELU(), nn.Linear(branch_dim, 1)
        )
        self.coop_head = CooperativeHead(
            ctx_dim=branch_dim * 3,
            hidden_dim=coop_hidden_dim,
            max_count=coop_max_count,
        )

    def forward(
        self,
        z_site: torch.Tensor,
        z_motif: torch.Tensor,
        z_path: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cond = condition.float()
        E_site = self.site_head(torch.cat([z_site, cond], dim=-1)).squeeze(-1)
        E_motif = self.motif_head(torch.cat([z_motif, cond], dim=-1)).squeeze(-1)
        E_path = self.path_head(torch.cat([z_path, cond], dim=-1)).squeeze(-1)

        n_h2o = cond[:, 0]
        n_co2 = cond[:, 1]
        ctx = torch.cat([z_site, z_motif, z_path], dim=-1)
        E_coop = self.coop_head(n_h2o, n_co2, ctx)
        E_pred = E_site + E_motif + E_path + E_coop
        return {
            "energy": E_pred,
            "E_pred": E_pred,
            "E_site": E_site,
            "E_motif": E_motif,
            "E_path": E_path,
            "E_coop": E_coop,
            "cooperative_pred": E_coop,
        }


@registry.register_model("escn_v9_causal_s1")
class EscnV9CausalS1(nn.Module):
    """V9 stage 1: eSCN + atom_type + masked atom recovery heads."""

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
        detach_atom_aux: bool = True,
        mask_token_atomic_number: int = 0,
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
        self.atom_type_head = AtomTypeHead(in_dim=sphere_channels, n_classes=atom_type_n_classes)
        self.mask_recon_head = nn.Sequential(
            nn.Linear(sphere_channels, atom_type_hidden_dim),
            nn.GELU(),
            nn.Linear(atom_type_hidden_dim, atom_type_n_classes),
        )
        self.detach_atom_aux = detach_atom_aux
        self.mask_token_atomic_number = mask_token_atomic_number

    @property
    def backbone(self):
        return self.base.backbone

    def forward(self, data) -> dict:
        emb = self.backbone.forward_embeddings(data)
        sphere_values = emb["sphere_values"]
        ns = int(self.backbone.num_sphere_samples)
        sc = int(self.backbone.sphere_channels)
        per_atom = sphere_values.view(-1, ns, sc).mean(dim=1)

        aux_feat = per_atom.detach() if self.detach_atom_aux else per_atom
        atom_type_logits = self.atom_type_head(aux_feat)

        recovery_mask = getattr(data, "atom_recovery_mask", None)
        if recovery_mask is not None and recovery_mask.any():
            masked_data = _clone_with_masked_atomic_numbers(
                data,
                recovery_mask.bool(),
                self.mask_token_atomic_number,
            )
            masked_emb = self.backbone.forward_embeddings(masked_data)
            masked_sphere_values = masked_emb["sphere_values"]
            masked_per_atom = masked_sphere_values.view(-1, ns, sc).mean(dim=1)
            recon_feat = masked_per_atom.detach() if self.detach_atom_aux else masked_per_atom
        else:
            recon_feat = aux_feat
        mask_recon_logits = self.mask_recon_head(recon_feat)

        out = self.base(data, emb=emb)
        out["atom_type_logits"] = atom_type_logits
        out["mask_recon_logits"] = mask_recon_logits
        if recovery_mask is not None:
            out["atom_recovery_mask"] = recovery_mask
        out["per_atom_features"] = per_atom
        return out


@registry.register_model("escn_v9_causal_s2")
class EscnV9CausalS2(nn.Module):
    """V9 stage 2: tri-branch representations + IRM-ready outputs."""

    def __init__(
        self,
        backbone: dict | None = None,
        reduce: str = "weighted_sum",
        weight_nn_hidden_dim: int = 64,
        use_initial_embeddings: bool = True,
        condition_dim: int = 2,
        condition_hidden_dim: int = 64,
        atom_type_n_classes: int = 100,
        branch_dim: int = 128,
        freeze_first_n_layers: int = 6,
        path_dropout_prob: float = 0.15,
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
        self.atom_type_head = AtomTypeHead(in_dim=sphere_channels, n_classes=atom_type_n_classes)
        self.tri_encoder = _TriBranchEncoder(in_dim=sphere_channels, branch_dim=branch_dim)
        # R-3: gas condition fused HERE (encoder is now gas-agnostic).
        # tri_energy_head input = [z_site, z_motif, z_path, condition]
        self.tri_energy_head = nn.Sequential(
            nn.Linear(branch_dim * 3 + condition_dim, branch_dim),
            nn.GELU(),
            nn.Linear(branch_dim, 1),
        )
        # env_infer_head still takes only structural features — that's the point:
        # if structure-only features can recover env, IRM has failed.
        self.env_infer_head = nn.Sequential(
            nn.Linear(branch_dim * 3, branch_dim), nn.GELU(), nn.Linear(branch_dim, 3)
        )
        self.freeze_first_n_layers = freeze_first_n_layers
        self.path_dropout_prob = path_dropout_prob

    @property
    def backbone(self):
        return self.base.backbone

    def freeze_s1_low_layers(self):
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

    def forward(self, data) -> dict:
        emb = self.backbone.forward_embeddings(data)
        sphere_values = emb["sphere_values"]
        ns = int(self.backbone.num_sphere_samples)
        sc = int(self.backbone.sphere_channels)
        per_atom = sphere_values.view(-1, ns, sc).mean(dim=1)

        out = self.base(data, emb=emb)
        atom_type_logits = self.atom_type_head(per_atom)

        n_graphs = int(data.natoms.shape[0]) if data.natoms.dim() > 0 else 1
        batch = data.batch
        condition = data.condition.float()
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)

        z_site, z_motif, z_path, z_path_aug = self.tri_encoder(
            per_atom=per_atom,
            atomic_numbers=data.atomic_numbers.long(),
            batch=batch,
            n_graphs=n_graphs,
            path_dropout_prob=self.path_dropout_prob,
        )
        env_logits = self.env_infer_head(torch.cat([z_site, z_motif, z_path], dim=-1))
        tri_energy = self.tri_energy_head(
            torch.cat([z_site, z_motif, z_path, condition], dim=-1)
        ).squeeze(-1)
        tri_energy_aug = self.tri_energy_head(
            torch.cat([z_site, z_motif, z_path_aug, condition], dim=-1)
        ).squeeze(-1)

        out.update(
            {
                "energy": tri_energy,
                "energy_aug": tri_energy_aug,
                "energy_base": out.get("energy"),
                "atom_type_logits": atom_type_logits,
                "per_atom_features": per_atom,
                "z_site": z_site,
                "z_motif": z_motif,
                "z_path": z_path,
                "z_path_aug": z_path_aug,
                "env_logits": env_logits,
            }
        )
        return out


@registry.register_model("escn_v9_causal_s3")
class EscnV9CausalS3(nn.Module):
    """V9 stage 3: mechanism decomposition + cooperative term."""

    def __init__(
        self,
        backbone: dict | None = None,
        reduce: str = "weighted_sum",
        weight_nn_hidden_dim: int = 64,
        use_initial_embeddings: bool = True,
        condition_dim: int = 2,
        condition_hidden_dim: int = 64,
        branch_dim: int = 128,
        coop_hidden_dim: int = 64,
        coop_max_count: float = 5.0,
        path_dropout_prob: float = 0.10,
        freeze_tri_encoder: bool = True,
        freeze_base: bool = True,
        **kwargs,
    ):
        super().__init__()
        # R-5: explicit freeze controls. Strict default = freeze both base and
        # tri_encoder so S3 only trains CausalMixer + CooperativeHead.
        # Set freeze_tri_encoder=False in yml for partial unfreeze (recommended
        # only if S2 metrics are weak and S3 cannot fit the energy target).
        self._cfg_freeze_tri_encoder = bool(freeze_tri_encoder)
        self._cfg_freeze_base = bool(freeze_base)
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
        self.tri_encoder = _TriBranchEncoder(in_dim=sphere_channels, branch_dim=branch_dim)

        self.causal_mixer = CausalMixer(
            branch_dim=branch_dim,
            coop_hidden_dim=coop_hidden_dim,
            coop_max_count=coop_max_count,
            condition_dim=condition_dim,
        )
        self.path_dropout_prob = path_dropout_prob

    @property
    def backbone(self):
        return self.base.backbone

    def freeze_s2_encoders(self):
        if self._cfg_freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
        if self._cfg_freeze_tri_encoder:
            for p in self.tri_encoder.parameters():
                p.requires_grad = False

    def forward(self, data) -> dict:
        emb = self.backbone.forward_embeddings(data)
        sphere_values = emb["sphere_values"]
        ns = int(self.backbone.num_sphere_samples)
        sc = int(self.backbone.sphere_channels)
        per_atom = sphere_values.view(-1, ns, sc).mean(dim=1)

        n_graphs = int(data.natoms.shape[0]) if data.natoms.dim() > 0 else 1
        batch = data.batch
        condition = data.condition.float()
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)

        z_site, z_motif, z_path, z_path_aug = self.tri_encoder(
            per_atom=per_atom,
            atomic_numbers=data.atomic_numbers.long(),
            batch=batch,
            n_graphs=n_graphs,
            path_dropout_prob=self.path_dropout_prob,
        )

        mix = self.causal_mixer(z_site, z_motif, z_path, condition)
        zero_motif = torch.zeros_like(z_motif)
        zero_path = torch.zeros_like(z_path)
        roll_motif = torch.roll(z_motif, shifts=1, dims=0).detach()
        roll_path = torch.roll(z_path, shifts=1, dims=0).detach()
        cf_motif_zero = self.causal_mixer(z_site, zero_motif, z_path, condition)
        cf_path_zero = self.causal_mixer(z_site, z_motif, zero_path, condition)
        cf_motif_swap = self.causal_mixer(z_site, roll_motif, z_path, condition)
        cf_path_swap = self.causal_mixer(z_site, z_motif, roll_path, condition)

        return {
            **mix,
            "energy_no_motif": cf_motif_zero["energy"],
            "energy_no_path": cf_path_zero["energy"],
            "energy_cf_motif_zero": cf_motif_zero["energy"],
            "energy_cf_path_zero": cf_path_zero["energy"],
            "energy_cf_motif_swap": cf_motif_swap["energy"],
            "energy_cf_path_swap": cf_path_swap["energy"],
            "cf_delta_motif_zero": cf_motif_zero["energy"] - mix["energy"],
            "cf_delta_path_zero": cf_path_zero["energy"] - mix["energy"],
            "cf_delta_motif_swap": cf_motif_swap["energy"] - mix["energy"],
            "cf_delta_path_swap": cf_path_swap["energy"] - mix["energy"],
            "z_site": z_site,
            "z_motif": z_motif,
            "z_path": z_path,
            "z_path_aug": z_path_aug,
            "per_atom_features": per_atom,
        }
