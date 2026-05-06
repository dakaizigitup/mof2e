"""
V7 Causal Trainer — supports all 3 stages of CMG.

Each stage has different loss components controlled by yml `causal:` block.

Stage 1:  L_total = L_energy + λ_atom · L_atom_type + λ_extensive · L_extensive
Stage 2:  L_total = L_energy + λ_atom · L_atom_type
                  + λ_inv · L_inv  + λ_motif · L_motif_consistency
Stage 3:  L_total = L_pred + λ_inv · L_inv + λ_S · L_S
                  + λ_coop · L_coop + λ_prior · L_α_prior + λ_reg · L_α_sparsity

Usage in yml:
    trainer: ocp_causal_v7
    causal:
      stage: 1                  # 1, 2, or 3
      lambda_atom_type: 0.1
      lambda_extensive: 0.01
      lambda_inv: 0.5           # stage 2/3
      lambda_motif: 0.1         # stage 2
      lambda_S: 1.0             # stage 3
      lambda_coop: 0.5          # stage 3
      lambda_prior: 0.5         # stage 3
      lambda_reg: 0.01          # stage 3
      init_from: /path/to/stage1_or_stage2_ckpt   # stage 2/3 only
      prior_alpha_path: /path/to/prior_alpha_per_atom.parquet  # stage 3
      cooperative_target_path: /path/to/cooperative_target.parquet  # stage 3
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fairchem.core.common.registry import registry
from fairchem.core.trainers.ocp_trainer import OCPTrainer

# Local causal-module imports
from causal_model.model_causal.causal_modules import (
    irmv1_penalty,
    loss_mask_prior, loss_mask_sparsity,
    loss_S_uninformative, loss_cooperative, loss_extensive,
)


@registry.register_trainer("ocp_causal_v7")
class CausalTrainerV7(OCPTrainer):
    """V7 trainer supporting all 3 stages."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Support both top-level (self.config["causal"]) and embedded
        # (self.config["model"]["causal"]) access patterns.
        cfg = self.config.get("causal", {})
        if not cfg:
            cfg = self.config.get("model", {}).get("causal", {})

        if not cfg:
            logging.warning("[CausalTrainerV7] No 'causal' block found in config — "
                            "trainer will fall back to plain ERM. Check yml has `causal:`")
        self.stage = int(cfg.get("stage", 1))
        self.lam_atom = float(cfg.get("lambda_atom_type", 0.1))
        self.lam_ext = float(cfg.get("lambda_extensive", 0.01))
        self.lam_inv = float(cfg.get("lambda_inv", 0.5))
        self.lam_motif = float(cfg.get("lambda_motif", 0.1))
        self.lam_S = float(cfg.get("lambda_S", 1.0))
        self.lam_coop = float(cfg.get("lambda_coop", 0.5))
        self.lam_prior = float(cfg.get("lambda_prior", 0.5))
        self.lam_reg = float(cfg.get("lambda_reg", 0.01))

        # Loud confirmation that V7 trainer is active
        logging.info(
            f"╔══════════════════════════════════════════════════════════╗\n"
            f"║  CausalTrainerV7 ACTIVE — stage={self.stage}                       ║\n"
            f"║  λ_atom_type={self.lam_atom}  λ_extensive={self.lam_ext}             ║\n"
            f"║  λ_inv={self.lam_inv}  λ_motif={self.lam_motif}  "
            f"λ_S={self.lam_S}  λ_coop={self.lam_coop}    ║\n"
            f"║  λ_prior={self.lam_prior}  λ_reg={self.lam_reg}                            ║\n"
            f"╚══════════════════════════════════════════════════════════╝"
        )

        # Auxiliaries (stage-specific)
        self._prior_alpha_lookup = None
        self._coop_target_lookup = None
        self._env_lookup = None

        if self.stage >= 2:
            env_path = cfg.get("env_labels_path",
                               "fairchem/causal_model/prep_data/env_labels.parquet")
            self._env_lookup = self._build_env_lookup(env_path)
            logging.info(f"[V7-S{self.stage}] env_labels: {len(self._env_lookup)} samples")

        if self.stage == 3:
            prior_path = cfg.get("prior_alpha_path",
                                 "fairchem/causal_model/prep_data/prior_alpha_per_atom.parquet")
            coop_path = cfg.get("cooperative_target_path",
                                "fairchem/causal_model/prep_data/cooperative_target.parquet")
            self._prior_alpha_lookup = self._build_prior_alpha_lookup(prior_path)
            self._coop_target_lookup = self._build_coop_target_lookup(coop_path)
            logging.info(f"[V7-S3] prior_α: {len(self._prior_alpha_lookup)} entries, "
                         f"coop targets: {len(self._coop_target_lookup)} samples")

        # If init_from is given, load ckpt weights (frozen by model.freeze_*)
        init_from = cfg.get("init_from", None)
        if init_from is not None and Path(init_from).is_file():
            logging.info(f"[V7-S{self.stage}] loading init_from {init_from}")
            sd = torch.load(init_from, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            logging.info(f"[V7-S{self.stage}] missing keys: {len(missing)}, "
                         f"unexpected: {len(unexpected)}")
            # Apply freezing if model exposes such methods
            mod = self.model.module if hasattr(self.model, "module") else self.model
            if self.stage == 2 and hasattr(mod, "freeze_s1_weights"):
                mod.freeze_s1_weights()
            if self.stage == 3 and hasattr(mod, "freeze_s2_weights"):
                mod.freeze_s2_weights()

    # ─────────────────────────────────────────────────────────
    # Auxiliary lookup builders
    # ─────────────────────────────────────────────────────────
    def _build_env_lookup(self, path):
        df = pd.read_parquet(path)
        # Key by (lmdb_name, n_h2o, n_co2) since lmdb_idx may differ between batches
        return {(r.lmdb_name, int(r.n_h2o), int(r.n_co2)): int(r.env_id)
                for r in df.itertuples()}

    def _build_prior_alpha_lookup(self, path):
        df = pd.read_parquet(path)
        # Key by (lmdb_name, atom_idx). Build per-name dict.
        # We need to map by name (since batch has data.name, not lmdb_idx directly).
        name_map = {}    # lmdb_name → np.ndarray of prior_α per atom
        for (split, idx), grp in df.groupby(["lmdb_split", "lmdb_idx"]):
            # We need lmdb_name. Look it up via separate query if not already.
            # For efficiency, also include lmdb_name column when available.
            pass
        # Re-key by lmdb_name lookup (need name mapping)
        # NOTE: prior_alpha_per_atom.parquet does NOT include lmdb_name column.
        # We'll reload with name from separate index.
        from pathlib import Path as _P
        idx_path = _P("/home/dell/autodl-tmp/lorafair") / \
                    "mechanism_analysis/outputs/A1_site_units/per_entry_metal_summary.parquet"
        idx_df = pd.read_parquet(idx_path)[["lmdb_split", "lmdb_idx", "lmdb_name"]]
        df2 = df.merge(idx_df, on=["lmdb_split", "lmdb_idx"], how="left")
        for name, grp in df2.groupby("lmdb_name"):
            arr = np.zeros(grp["atom_idx"].max() + 1, dtype=np.float32)
            arr[grp["atom_idx"].values] = grp["prior_alpha"].values.astype(np.float32)
            name_map[str(name)] = arr
        return name_map

    def _build_coop_target_lookup(self, path):
        df = pd.read_parquet(path)
        return {(r.lmdb_name, int(r.n_h2o), int(r.n_co2)):
                (float(r.cooperative_target) if pd.notna(r.cooperative_target) else None)
                for r in df.itertuples()}

    # ─────────────────────────────────────────────────────────
    # Forward — attach prior_alpha to batch if stage 3
    # ─────────────────────────────────────────────────────────
    def _forward(self, batch):
        if self.stage == 3 and self._prior_alpha_lookup is not None:
            names = batch.name if isinstance(batch.name, (list, tuple)) else [batch.name]
            n_per = batch.natoms.tolist() if hasattr(batch.natoms, "tolist") else [int(batch.natoms)]
            prior_list = []
            for nm, n in zip(names, n_per):
                base = str(nm)
                arr = self._prior_alpha_lookup.get(base, None)
                if arr is None:
                    arr = np.full(n, 0.3, dtype=np.float32)
                else:
                    if len(arr) < n:
                        arr = np.concatenate([arr, np.full(n - len(arr), 0.3, dtype=np.float32)])
                    elif len(arr) > n:
                        arr = arr[:n]
                prior_list.append(arr)
            batch.prior_alpha_per_atom = torch.from_numpy(
                np.concatenate(prior_list)).to(self.device)

        # Now call the model directly (we need raw `out` dict, not OCPTrainer's
        # output reformatting because we need atom_type_logits etc.)
        out = self.model(batch.to(self.device))
        # OCPTrainer expects a dict with target keys; ensure "energy" present
        return out

    # ─────────────────────────────────────────────────────────
    # Loss computation
    # ─────────────────────────────────────────────────────────
    def _compute_loss(self, out, batch) -> torch.Tensor:
        # Base energy loss (MAE on normalized energy)
        target = batch["energy"].view(batch.natoms.numel(), -1).to(self.device)
        if "energy" in self.normalizers:
            target_norm = self.normalizers["energy"].norm(target)
        else:
            target_norm = target

        E_pred = out["energy"].view(batch.natoms.numel(), -1)
        loss_energy = F.l1_loss(E_pred, target_norm)
        total = loss_energy

        # Stage 1+: atom_type loss
        if self.stage >= 1 and "atom_type_logits" in out:
            atom_z = batch.atomic_numbers.long().to(self.device)
            logits = out["atom_type_logits"]
            # Clamp z to valid range
            z_clip = atom_z.clamp(0, logits.size(-1) - 1)
            loss_atom = F.cross_entropy(logits, z_clip)
            total = total + self.lam_atom * loss_atom

        # Stage 1: extensive penalty
        # NOTE: this loss is DISABLED by default (lam_ext=0). It used to compare
        # normalized E_pred against unnormalized expected, causing divergence.
        # Kept here in case future stages want a properly-scaled extensive prior.
        if self.stage == 1 and self.lam_ext > 0:
            n_atoms = batch.natoms.float().to(self.device)
            loss_ext = loss_extensive(E_pred.squeeze(-1), n_atoms)
            total = total + self.lam_ext * loss_ext

        # Stage 2/3: IRM invariance
        if self.stage >= 2 and self.lam_inv > 0 and self.model.training:
            env_ids = self._get_env_ids(batch)
            n_envs = 3
            if env_ids is not None and (env_ids >= 0).any():
                pen = irmv1_penalty(E_pred.squeeze(-1), target_norm.squeeze(-1),
                                    env_ids, n_envs=n_envs)
                total = total + self.lam_inv * pen

        # Stage 3: dual-head losses
        if self.stage == 3:
            # L_S: context branch should predict prior (mean energy)
            if "energy_context" in out:
                # Use batch mean as crude prior
                E_S = out["energy_context"].squeeze(-1)
                E_prior = target_norm.squeeze(-1).mean().expand_as(E_S)
                total = total + self.lam_S * loss_S_uninformative(E_S, E_prior)

            # L_α prior anchor
            if "alpha" in out and hasattr(batch, "prior_alpha_per_atom"):
                alpha = out["alpha"]
                prior = batch.prior_alpha_per_atom
                total = total + self.lam_prior * loss_mask_prior(alpha, prior)
                total = total + self.lam_reg * loss_mask_sparsity(alpha)

            # L_coop
            if "cooperative_pred" in out and self._coop_target_lookup is not None:
                coop_pred = out["cooperative_pred"]
                names = batch.name if isinstance(batch.name, (list, tuple)) else [batch.name]
                cond = batch.condition.view(-1, 2).to(self.device)
                targets = []
                masks = []
                for i, (nm, c) in enumerate(zip(names, cond)):
                    h = int(c[0].item()); cc = int(c[1].item())
                    t = self._coop_target_lookup.get((str(nm), h, cc))
                    if t is None:
                        targets.append(0.0); masks.append(False)
                    else:
                        targets.append(t); masks.append(True)
                tgt = torch.tensor(targets, dtype=coop_pred.dtype, device=coop_pred.device)
                msk = torch.tensor(masks, dtype=torch.bool, device=coop_pred.device)
                total = total + self.lam_coop * loss_cooperative(coop_pred, tgt, msk)

        return total

    def _get_env_ids(self, batch):
        if self._env_lookup is None:
            return None
        names = batch.name if isinstance(batch.name, (list, tuple)) else [batch.name]
        cond = batch.condition.view(-1, 2)
        ids = []
        for nm, c in zip(names, cond):
            h = int(c[0].item()); cc = int(c[1].item())
            ids.append(self._env_lookup.get((str(nm), h, cc), -1))
        return torch.tensor(ids, dtype=torch.long, device=self.device)
