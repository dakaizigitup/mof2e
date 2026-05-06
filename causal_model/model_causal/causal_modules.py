"""
Shared causal modules for V7 (CMG) and V8 (AFCMG).

Components
----------
AtomMaskHead         : per-atom × per-channel mask α ∈ [0,1]^(N, C),
                       only acts on l=0 invariant channel of SO(3) embedding
DualEnergyHead       : two parallel energy heads (causal / context)
CooperativeHead      : ψ(n_h2o, n_co2) with hard-zero constraint
                       on pure conditions (V8 analytical form available)
IRMv1Penalty         : standard IRM penalty across environments
                       (used in V7 stage 2/3 and V8 stage 2/3)
AtomTypeHead         : auxiliary classifier for atom_type supervision
                       (Stage 1 of both V7 and V8)

Key design decisions
--------------------
1. Mask α: shape (N, C), per-atom × per-channel. Multiplies l=0 channel only.
   Higher l channels (l>=1) pass through unchanged → SO(3) equivariance preserved.

2. AtomMaskHead input: l=0 invariant features + prior_α + neighbor element
   summary. Output: σ(MLP(...)) ∈ [0,1]^C.

3. CooperativeHead with chemistry hard floor: ψ = sigmoid(MLP(h, c)) * h * c
   * scale(MLP_a(h,c)). When h=0 or c=0, ψ=0 by construction (no parameters
   needed for the cooperative_zero loss; it's architectural).

4. All modules avoid in-place ops on shared tensors (clone before modify).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# 1. AtomMaskHead
# ─────────────────────────────────────────────────────────────────────
class AtomMaskHead(nn.Module):
    """Outputs per-atom × per-channel soft mask α ∈ [0,1]^(N, C)."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 64,
                 use_prior: bool = True, n_neighbor_types: int = 5):
        super().__init__()
        # Input: l0_features (in_dim) + prior_α (in_dim if per-channel, else 1)
        # + neighbor_summary (n_neighbor_types) + atom_z_onehot (10, optional)
        extra = 0
        if use_prior:
            extra += in_dim   # prior is per-channel (we broadcast scalar prior to channels)
        extra += n_neighbor_types
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + extra, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
            nn.Sigmoid(),
        )
        self.use_prior = use_prior
        # Initialize to identity-ish: output ≈ prior_α when prior is given,
        # ≈ 0.5 otherwise. This is achieved by zero-init the last linear layer.
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.zeros_(self.mlp[-2].bias)

    def forward(
        self,
        l0_features: torch.Tensor,           # (N, C)
        prior_alpha: torch.Tensor | None,    # (N,) scalar prior, or None
        neighbor_summary: torch.Tensor,      # (N, n_neighbor_types)
    ) -> torch.Tensor:
        N, C = l0_features.shape
        feats = [l0_features]
        if self.use_prior:
            if prior_alpha is None:
                prior_broadcast = torch.full((N, C), 0.5, device=l0_features.device,
                                              dtype=l0_features.dtype)
            else:
                # broadcast scalar prior to channels
                prior_broadcast = prior_alpha.view(N, 1).expand(-1, C).contiguous()
            feats.append(prior_broadcast)
        feats.append(neighbor_summary)
        x = torch.cat(feats, dim=-1)
        residual = self.mlp(x)              # (N, C), values around 0.5 due to zero-init

        # Final α: blend prior + learned residual; if prior given, default towards prior
        if self.use_prior and prior_alpha is not None:
            base = prior_alpha.view(N, 1).expand(-1, C)
            alpha = 0.5 * base + 0.5 * residual
        else:
            alpha = residual
        return alpha.clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────
# 2. DualEnergyHead
# ─────────────────────────────────────────────────────────────────────
class DualEnergyHead(nn.Module):
    """Two parallel energy heads operating on causal/context branches.

    Each head is a thin MLP: (per-atom features) → per-atom energy.
    Total energy = sum over atoms (with optional weighting).
    """
    def __init__(self, in_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        def make_head():
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        self.causal_head = make_head()
        self.context_head = make_head()

    def forward(
        self,
        h_causal: torch.Tensor,    # (N, C)
        h_context: torch.Tensor,   # (N, C)
        batch: torch.Tensor,       # (N,) graph index per atom
        n_graphs: int,
    ):
        node_E_C = self.causal_head(h_causal).squeeze(-1)       # (N,)
        node_E_S = self.context_head(h_context).squeeze(-1)     # (N,)

        E_C = torch.zeros(n_graphs, device=h_causal.device, dtype=h_causal.dtype)
        E_S = torch.zeros_like(E_C)
        E_C.index_add_(0, batch, node_E_C)
        E_S.index_add_(0, batch, node_E_S)
        return E_C, E_S, node_E_C, node_E_S


# ─────────────────────────────────────────────────────────────────────
# 3. CooperativeHead
# ─────────────────────────────────────────────────────────────────────
class CooperativeHead(nn.Module):
    """
    ψ(n_h2o, n_co2) with hard chemistry constraint:
        ψ(0, c) = 0  ∀ c
        ψ(h, 0) = 0  ∀ h

    Achieved analytically by multiplying by h*c, so the cooperative_zero loss
    is satisfied by construction (no explicit penalty needed).

    Form: ψ = (h * c / max_count²) * sigmoid(MLP_gate(h, c)) * MLP_amp(h, c, ctx)
    """
    def __init__(self, ctx_dim: int = 64, hidden_dim: int = 32, max_count: float = 5.0):
        super().__init__()
        self.max_count = max_count
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )
        self.amp_mlp = nn.Sequential(
            nn.Linear(2 + ctx_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        n_h2o: torch.Tensor,    # (B,)
        n_co2: torch.Tensor,    # (B,)
        ctx: torch.Tensor,      # (B, ctx_dim)
    ) -> torch.Tensor:           # (B,) cooperative term
        h = n_h2o.float()
        c = n_co2.float()
        # Hard zero on pure conditions: prefactor h*c
        prefactor = (h * c) / (self.max_count ** 2)   # (B,)
        cond = torch.stack([h, c], dim=-1)            # (B, 2)
        gate = self.gate_mlp(cond).squeeze(-1)        # (B,)  ∈ [0,1]
        amp_input = torch.cat([cond, ctx], dim=-1)
        amp = self.amp_mlp(amp_input).squeeze(-1)     # (B,)
        return prefactor * gate * amp                 # (B,) — equals 0 when h=0 or c=0


# ─────────────────────────────────────────────────────────────────────
# 4. IRMv1 Penalty
# ─────────────────────────────────────────────────────────────────────
def irmv1_penalty(predictions: torch.Tensor, targets: torch.Tensor,
                  env_ids: torch.Tensor, n_envs: int = 3) -> torch.Tensor:
    """Standard IRMv1 penalty (Arjovsky et al. 2019) over discrete environments.

    For each environment e, compute squared gradient of L_e w.r.t. dummy weight
    w (initialized at 1.0). Sum across environments.
    """
    dummy_w = torch.tensor(1.0, requires_grad=True, device=predictions.device)
    penalty = predictions.new_zeros(())
    for e in range(n_envs):
        mask = (env_ids == e)
        if not mask.any():
            continue
        loss_e = F.mse_loss(predictions[mask] * dummy_w, targets[mask])
        grad_e = torch.autograd.grad(loss_e, [dummy_w],
                                      create_graph=True, retain_graph=True)[0]
        penalty = penalty + grad_e.pow(2)
    return penalty


# ─────────────────────────────────────────────────────────────────────
# 5. AtomTypeHead
# ─────────────────────────────────────────────────────────────────────
class AtomTypeHead(nn.Module):
    """Auxiliary atom-type classifier; supervised by ground-truth atomic_number.
    Used in Stage 1 to ensure atom encoder learns clear element-aware features.
    """
    def __init__(self, in_dim: int = 128, n_classes: int = 100):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, l0_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(l0_features)


# ─────────────────────────────────────────────────────────────────────
# 6. Loss helpers
# ─────────────────────────────────────────────────────────────────────
def loss_mask_prior(alpha: torch.Tensor, prior_alpha: torch.Tensor) -> torch.Tensor:
    """L_prior = ‖α - prior_α‖²  (per-atom × per-channel anchor)."""
    if prior_alpha.dim() == 1:
        prior_alpha = prior_alpha.unsqueeze(-1).expand_as(alpha)
    return F.mse_loss(alpha, prior_alpha)


def loss_mask_sparsity(alpha: torch.Tensor) -> torch.Tensor:
    """L_reg = mean ‖α‖_1."""
    return alpha.abs().mean()


def loss_S_uninformative(E_S: torch.Tensor, prior_E: torch.Tensor) -> torch.Tensor:
    """L_S = ‖E_S - E_prior‖²  (force context branch to predict the prior)."""
    return F.mse_loss(E_S, prior_E)


def loss_cooperative(coop_pred: torch.Tensor, coop_target: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """L_coop = MSE on samples where target is non-NaN (mask=True)."""
    if not mask.any():
        return coop_pred.new_zeros(())
    return F.mse_loss(coop_pred[mask], coop_target[mask])


def loss_extensive(E_pred: torch.Tensor, n_atoms: torch.Tensor,
                   per_atom_mean: float = -0.005) -> torch.Tensor:
    """L_extensive: weak penalty enforcing E ≈ N_atoms × per_atom_mean."""
    expected = n_atoms.float() * per_atom_mean
    return F.mse_loss(E_pred, expected) * 0.01     # weak


def neighbor_summary_simple(atomic_numbers: torch.Tensor,
                             edge_index: torch.Tensor,
                             n_categories: int = 5) -> torch.Tensor:
    """Compute per-atom neighbor element category counts.

    Categories: [O, N, C, metal, other]
    Returns (N, n_categories) normalized counts.
    """
    is_O = (atomic_numbers == 8)
    is_N = (atomic_numbers == 7)
    is_C = (atomic_numbers == 6)
    # rough metal mask: Z >= 11 and not in {15, 16, 17, 35, 53}
    main_nonmetal = torch.tensor([15, 16, 17, 35, 53], device=atomic_numbers.device)
    is_metal = (atomic_numbers >= 11) & ~torch.isin(atomic_numbers, main_nonmetal)
    is_metal = is_metal & ~(is_O | is_N | is_C)
    is_other = ~(is_O | is_N | is_C | is_metal)

    src = edge_index[0]   # source atom indices
    dst = edge_index[1]   # destination atom indices
    N = atomic_numbers.size(0)
    out = torch.zeros((N, n_categories), device=atomic_numbers.device,
                      dtype=torch.float32)
    cats = [is_O, is_N, is_C, is_metal, is_other]
    for k, cat in enumerate(cats):
        cat_src = cat.float()[src]    # 1 if source is in this category
        out[:, k].index_add_(0, dst, cat_src)

    deg = out.sum(dim=1, keepdim=True).clamp(min=1)
    return out / deg
