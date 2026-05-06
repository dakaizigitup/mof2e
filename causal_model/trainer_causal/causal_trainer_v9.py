"""
V9 Causal Trainer — supports stage 1/2/3.

Stage 1:
  L = L_energy + λ_type * L_atom_type + λ_mask * L_mask_recon
Stage 2:
  L = L_energy + λ_inv * L_IRM + λ_dec * L_decorrel + λ_sub * L_subgraph_consistency
Stage 3:
  L = L_energy + λ_cf * L_counterfactual + λ_phys * L_physics + λ_inv * L_IRM
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from fairchem.core.common.registry import registry
from fairchem.core.trainers.ocp_trainer import OCPTrainer
from causal_model.model_causal.causal_modules import irmv1_penalty, loss_extensive


@registry.register_trainer("ocp_causal_v9")
class CausalTrainerV9(OCPTrainer):
    """V9 trainer for staged causal embedding."""

    _CKPT_CRITICAL_PREFIXES = {
        2: (
            "base.backbone.sphere_embedding",
            "base.backbone.layer_blocks.0",
        ),
        3: (
            "base.backbone.sphere_embedding",
            "base.backbone.layer_blocks.0",
            "tri_encoder",
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cfg = self.config.get("causal", {})
        if not cfg:
            cfg = self.config.get("model", {}).get("causal", {})

        self.stage = int(cfg.get("stage", 1))
        self.lam_type = float(cfg.get("lambda_type", cfg.get("lambda_atom_type", 0.05)))
        self.lam_mask = float(cfg.get("lambda_mask", 0.05))
        self.mask_ratio = float(cfg.get("mask_ratio", 0.15))
        self.mask_token_atomic_number = int(cfg.get("mask_token_atomic_number", 0))

        self.lam_inv = float(cfg.get("lambda_inv", 0.0))
        self.lam_dec = float(cfg.get("lambda_dec", 0.0))
        self.lam_sub = float(cfg.get("lambda_sub", 0.0))
        self.lam_env = float(cfg.get("lambda_env", 0.0))

        self.lam_cf = float(cfg.get("lambda_cf", 0.0))
        self.lam_phys = float(cfg.get("lambda_phys", 0.0))
        self.cf_min_effect = float(cfg.get("cf_min_effect", 0.01))
        self.cf_motif_direction = float(cfg.get("cf_motif_direction", 1.0))
        self.cf_path_direction = float(cfg.get("cf_path_direction", 1.0))
        self.ext_per_atom_mean = float(cfg.get("ext_per_atom_mean", -0.005))
        self.lam_phys_coop_zero = float(cfg.get("lambda_phys_coop_zero", 1.0))
        self.lam_phys_ext = float(cfg.get("lambda_phys_ext", 0.2))
        self.lam_phys_vol = float(cfg.get("lambda_phys_vol", 0.2))

        logging.info(
            "[CausalTrainerV9] ACTIVE "
            f"stage={self.stage}, λ_type={self.lam_type}, λ_mask={self.lam_mask}, "
            f"λ_inv={self.lam_inv}, λ_dec={self.lam_dec}, λ_sub={self.lam_sub}, λ_env={self.lam_env}, "
            f"λ_cf={self.lam_cf}, λ_phys={self.lam_phys}"
        )

        # P-5: track how often the cf_*_direction batch field is present vs.
        # we fell back to the global config direction. Reported every N steps
        # so we can immediately see if the dataset actually carries directions.
        self._cf_dir_log_every = int(cfg.get("cf_dir_log_every", 100))
        self._cf_dir_steps = 0
        self._cf_dir_motif_fallback = 0
        self._cf_dir_motif_total = 0
        self._cf_dir_path_fallback = 0
        self._cf_dir_path_total = 0

        mod = self.model.module if hasattr(self.model, "module") else self.model
        init_from = cfg.get("init_from", None)
        if init_from is not None and Path(init_from).is_file():
            logging.info(f"[V9-S{self.stage}] loading init_from {init_from}")
            sd = torch.load(init_from, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            logging.info(
                f"[V9-S{self.stage}] missing keys: {len(missing)}, unexpected: {len(unexpected)}"
            )
            self._check_loaded_critical_weights(sd, missing)
        elif init_from is not None:
            logging.warning(
                f"[V9-S{self.stage}] init_from does not exist: {init_from}. "
                "Continuing with current model weights."
            )

        if self.stage == 2 and hasattr(mod, "freeze_s1_low_layers"):
            mod.freeze_s1_low_layers()
            logging.info("[V9-S2] freeze_s1_low_layers applied")
        if self.stage == 3 and hasattr(mod, "freeze_s2_encoders"):
            mod.freeze_s2_encoders()
            logging.info("[V9-S3] freeze_s2_encoders applied")

    def _check_loaded_critical_weights(self, state_dict: dict, missing: list[str]) -> None:
        prefixes = self._CKPT_CRITICAL_PREFIXES.get(self.stage, ())
        if not prefixes:
            return

        # Stage 3 freezes tri_encoder right after this check; if tri_encoder
        # weights did not actually load from S2 ckpt, S3 would silently train
        # on a frozen *random* encoder. Treat that as a hard error.
        hard_required = self.stage == 3

        missing_set = set(missing)
        failures: list[str] = []
        for prefix in prefixes:
            found_in_ckpt = any(k.startswith(prefix) for k in state_dict)
            still_missing = any(k.startswith(prefix) for k in missing_set)
            if not found_in_ckpt:
                msg = f"checkpoint has no keys under critical prefix '{prefix}'"
                if hard_required:
                    failures.append(msg)
                else:
                    logging.warning(f"[V9-S{self.stage}] {msg}")
            elif still_missing:
                msg = f"some critical model keys remain missing under '{prefix}'"
                if hard_required:
                    failures.append(msg)
                else:
                    logging.warning(f"[V9-S{self.stage}] {msg}")
            else:
                logging.info(f"[V9-S{self.stage}] critical prefix loaded: {prefix}")

        if failures:
            raise RuntimeError(
                f"[V9-S{self.stage}] critical-weight check failed (init_from would "
                f"leave frozen modules randomly initialized): "
                + "; ".join(failures)
            )

    def _forward(self, batch):
        if self.stage == 1 and self.lam_mask > 0 and self.model.training:
            atom_z = batch.atomic_numbers
            recovery_mask = torch.rand(atom_z.shape, device=atom_z.device) < self.mask_ratio
            if not recovery_mask.any() and atom_z.numel() > 0:
                recovery_mask[torch.randint(atom_z.numel(), (1,), device=atom_z.device)] = True
            batch.atom_recovery_mask = recovery_mask
            batch.mask_token_atomic_number = self.mask_token_atomic_number
        return self.model(batch.to(self.device))

    @staticmethod
    def _build_env_ids_from_condition(condition: torch.Tensor) -> torch.Tensor:
        h = condition[:, 0]
        c = condition[:, 1]
        env = torch.full((condition.size(0),), 2, device=condition.device, dtype=torch.long)  # mixed
        env[(h == 0) & (c > 0)] = 0  # pure CO2
        env[(h > 0) & (c == 0)] = 1  # pure H2O
        return env

    @staticmethod
    def _corr_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x - x.mean()
        y = y - y.mean()
        den = (x.std(unbiased=False) * y.std(unbiased=False)).clamp(min=1e-8)
        return ((x * y).mean() / den).pow(2)

    def _decorrel_loss(self, z_site: torch.Tensor, z_motif: torch.Tensor, z_path: torch.Tensor) -> torch.Tensor:
        def pair(a, b):
            a0 = a - a.mean(dim=0, keepdim=True)
            b0 = b - b.mean(dim=0, keepdim=True)
            num = (a0 * b0).mean(dim=0)
            den = (a0.std(dim=0, unbiased=False) * b0.std(dim=0, unbiased=False)).clamp(min=1e-8)
            return (num / den).pow(2).mean()

        return pair(z_site, z_motif) + pair(z_site, z_path) + pair(z_motif, z_path)

    def _cf_direction_tensor(
        self, batch, key: str, fallback: float, ref: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """Returns (direction_tensor, used_fallback)."""
        raw = getattr(batch, key, None)
        if raw is None and hasattr(batch, "__getitem__"):
            try:
                raw = batch[key]
            except Exception:
                raw = None

        if raw is None:
            return torch.full_like(ref, float(fallback)), True

        direction = torch.as_tensor(raw, dtype=ref.dtype, device=ref.device).view(-1)
        if direction.numel() == 1:
            return direction.expand_as(ref), False
        if direction.numel() != ref.numel():
            logging.warning(
                f"[V9-S3] {key} has {direction.numel()} values, expected {ref.numel()}; "
                "falling back to config direction."
            )
            return torch.full_like(ref, float(fallback)), True
        return direction, False

    def _log_cf_direction_usage(self, n_samples: int, m_fb: bool, p_fb: bool) -> None:
        self._cf_dir_steps += 1
        self._cf_dir_motif_total += n_samples
        self._cf_dir_path_total += n_samples
        if m_fb:
            self._cf_dir_motif_fallback += n_samples
        if p_fb:
            self._cf_dir_path_fallback += n_samples
        if self._cf_dir_steps % max(1, self._cf_dir_log_every) == 0:
            tot_m = max(1, self._cf_dir_motif_total)
            tot_p = max(1, self._cf_dir_path_total)
            logging.info(
                f"[V9-S3] cf_direction fallback ratio "
                f"motif={self._cf_dir_motif_fallback}/{tot_m}={self._cf_dir_motif_fallback/tot_m:.1%}, "
                f"path={self._cf_dir_path_fallback}/{tot_p}={self._cf_dir_path_fallback/tot_p:.1%}"
            )

    def _compute_loss(self, out, batch) -> torch.Tensor:
        target = batch["energy"].view(batch.natoms.numel(), -1).to(self.device)
        if "energy" in self.normalizers:
            target_norm = self.normalizers["energy"].norm(target)
        else:
            target_norm = target

        pred = out["energy"].view(batch.natoms.numel(), -1)
        loss_energy = F.l1_loss(pred, target_norm)
        total = loss_energy

        condition = batch.condition.view(-1, 2).to(self.device)

        if self.stage == 1:
            atom_z = batch.atomic_numbers.long().to(self.device)
            if "atom_type_logits" in out:
                logits = out["atom_type_logits"]
                z_clip = atom_z.clamp(0, logits.size(-1) - 1)
                loss_type = F.cross_entropy(logits, z_clip)
                total = total + self.lam_type * loss_type

            if "mask_recon_logits" in out:
                recon_logits = out["mask_recon_logits"]
                z_clip_recon = atom_z.clamp(0, recon_logits.size(-1) - 1)
                mask = out.get("atom_recovery_mask", None)
                if mask is None:
                    mask = getattr(batch, "atom_recovery_mask", None)
                if mask is None:
                    mask = torch.ones_like(atom_z, dtype=torch.bool)
                mask = mask.to(self.device).bool()
                if mask.any():
                    loss_mask = F.cross_entropy(recon_logits[mask], z_clip_recon[mask])
                    total = total + self.lam_mask * loss_mask

            return total

        env_ids = self._build_env_ids_from_condition(condition)

        # R-2: IRM is computed only over pure_CO2 (env=0) and pure_H2O (env=1).
        # `mixed` (env=2) carries the cooperative effect that S3's coop head is
        # *supposed* to model — including it in IRM would force S2 to erase the
        # very mechanism S3 must learn.
        # R-1 (defensive): IRM penalty needs ≥2 distinct envs to be meaningful.
        # If a batch happens to have only one of {pure_CO2, pure_H2O}, skip.
        if self.stage >= 2 and self.lam_inv > 0 and self.model.training:
            irm_mask = (env_ids == 0) | (env_ids == 1)
            n_env0 = int((env_ids == 0).sum().item())
            n_env1 = int((env_ids == 1).sum().item())
            if n_env0 > 0 and n_env1 > 0:
                loss_inv = irmv1_penalty(
                    predictions=pred.squeeze(-1)[irm_mask],
                    targets=target_norm.squeeze(-1)[irm_mask],
                    env_ids=env_ids[irm_mask],
                    n_envs=2,
                )
                total = total + self.lam_inv * loss_inv
            else:
                self._irm_skip_count = getattr(self, "_irm_skip_count", 0) + 1
                if self._irm_skip_count % 200 == 1:
                    logging.info(
                        f"[V9-S{self.stage}] IRM skipped (env0={n_env0}, env1={n_env1}); "
                        f"total skips so far: {self._irm_skip_count}. "
                        "Increase batch_size or enable env-balanced sampling."
                    )

        if self.stage >= 2 and self.lam_dec > 0 and all(k in out for k in ["z_site", "z_motif", "z_path"]):
            loss_dec = self._decorrel_loss(out["z_site"], out["z_motif"], out["z_path"])
            total = total + self.lam_dec * loss_dec

        if self.stage >= 2 and self.lam_sub > 0 and "energy_aug" in out:
            loss_sub = F.mse_loss(out["energy"], out["energy_aug"])
            total = total + self.lam_sub * loss_sub

        if self.stage == 2 and self.lam_env > 0 and "env_logits" in out:
            loss_env = F.cross_entropy(out["env_logits"], env_ids)
            total = total + self.lam_env * loss_env

        if self.stage == 3:
            if self.lam_cf > 0 and all(k in out for k in ["cf_delta_motif_zero", "cf_delta_path_zero"]):
                mixed = env_ids == 2
                if mixed.any():
                    motif_dir, m_fb = self._cf_direction_tensor(
                        batch,
                        "cf_motif_direction",
                        self.cf_motif_direction,
                        out["cf_delta_motif_zero"],
                    )
                    path_dir, p_fb = self._cf_direction_tensor(
                        batch,
                        "cf_path_direction",
                        self.cf_path_direction,
                        out["cf_delta_path_zero"],
                    )
                    if self.model.training:
                        self._log_cf_direction_usage(int(mixed.sum().item()), m_fb, p_fb)
                    loss_cf_m = F.relu(
                        self.cf_min_effect
                        - motif_dir[mixed] * out["cf_delta_motif_zero"][mixed]
                    ).mean()
                    loss_cf_p = F.relu(
                        self.cf_min_effect
                        - path_dir[mixed] * out["cf_delta_path_zero"][mixed]
                    ).mean()
                    total = total + self.lam_cf * (loss_cf_m + loss_cf_p)

            if self.lam_phys > 0:
                phys = pred.new_zeros(())

                if "cooperative_pred" in out:
                    coop = out["cooperative_pred"]
                    pure = env_ids != 2
                    if pure.any():
                        phys = phys + self.lam_phys_coop_zero * F.mse_loss(
                            coop[pure],
                            torch.zeros_like(coop[pure]),
                        )

                pred_denorm = pred.squeeze(-1)
                if "energy" in self.normalizers:
                    pred_denorm = self.normalizers["energy"].denorm(pred).squeeze(-1)
                n_atoms = batch.natoms.float().to(self.device)
                phys = phys + self.lam_phys_ext * loss_extensive(
                    pred_denorm,
                    n_atoms,
                    per_atom_mean=self.ext_per_atom_mean,
                )

                if hasattr(batch, "cell") and batch.cell is not None:
                    cell = batch.cell.to(self.device)
                    if cell.dim() == 3 and cell.size(0) == pred_denorm.size(0):
                        vol = torch.abs(torch.linalg.det(cell))
                        phys = phys + self.lam_phys_vol * self._corr_sq(pred_denorm, vol)

                total = total + self.lam_phys * phys

        return total
