"""
Five evaluation metrics for V7/V8 causal-embedded models.

Per design doc:
  1. In-distribution MAE      (vs ERM baseline)
  2. OOD MAE                  (on hold-out topology / metal subsets)
  3. Mask-α IoU vs A4 prior   (V7 only) — does trained mask agree with mechanism prior?
  4. Cooperative recovery     — Pearson r vs B3.1 cooperative target
  5. Counterfactual recovery  — Pearson r vs D3 free-COOH removal target

All metrics are unit-tested and produce a unified report dict that compares
V7 / V8 / ERM_baseline side-by-side.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path("/home/dell/autodl-tmp/lorafair")
PREP = REPO / "fairchem/causal_model/prep_data"


# ─────────────────────────────────────────────────────────────────────
# 1. MAE on a given subset
# ─────────────────────────────────────────────────────────────────────
def mae(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]
    if len(predictions) == 0:
        return float("nan")
    return float(np.mean(np.abs(predictions - targets)))


def mae_id_and_ood(
    predictions: pd.DataFrame,            # columns: lmdb_split, lmdb_idx, energy_pred, energy_target
    ood_split_path: Path = PREP / "ood_split.json",
) -> Dict[str, float]:
    """Returns MAE on:
      in_distribution_train, ood_topology, ood_metal, ood_combined, standard_val
    """
    splits = json.loads(Path(ood_split_path).read_text())
    out: Dict[str, float] = {}

    for split_name in ("in_distribution_train", "ood_topology",
                       "ood_metal", "ood_combined"):
        idxs = set(splits[split_name]["lmdb_idxs"])
        sub = predictions[
            (predictions["lmdb_split"] == "train") &
            (predictions["lmdb_idx"].isin(idxs))
        ]
        out[f"mae_{split_name}"] = mae(
            sub["energy_pred"].values, sub["energy_target"].values)

    val_sub = predictions[predictions["lmdb_split"] == "val"]
    out["mae_standard_val"] = mae(
        val_sub["energy_pred"].values, val_sub["energy_target"].values)
    return out


# ─────────────────────────────────────────────────────────────────────
# 2. Mask-α IoU vs A4 prior  (V7 only)
# ─────────────────────────────────────────────────────────────────────
def mask_alpha_iou(
    trained_alpha: pd.DataFrame,   # columns: lmdb_split, lmdb_idx, atom_idx, alpha
    threshold_trained: float = 0.5,
    threshold_prior: float = 0.7,
    prior_path: Path = PREP / "prior_alpha_per_atom.parquet",
) -> Dict[str, float]:
    """IoU between (trained α > thresh_t) and (prior_α > thresh_p)."""
    prior = pd.read_parquet(prior_path)
    df = prior.merge(
        trained_alpha[["lmdb_split", "lmdb_idx", "atom_idx", "alpha"]],
        on=["lmdb_split", "lmdb_idx", "atom_idx"], how="left")
    df["alpha"] = df["alpha"].fillna(0.0)

    trained_pos = df["alpha"] > threshold_trained
    prior_pos = df["prior_alpha"] > threshold_prior
    intersection = (trained_pos & prior_pos).sum()
    union = (trained_pos | prior_pos).sum()
    iou = float(intersection / union) if union > 0 else float("nan")
    pearson = float(pearsonr(df["alpha"].values, df["prior_alpha"].values)[0])
    return dict(
        mask_iou=iou,
        mask_pearson=pearson,
        n_trained_pos=int(trained_pos.sum()),
        n_prior_pos=int(prior_pos.sum()),
        n_intersection=int(intersection),
    )


# ─────────────────────────────────────────────────────────────────────
# 3. Cooperative recovery  (V7/V8)
# ─────────────────────────────────────────────────────────────────────
def cooperative_recovery(
    predicted_cooperative: pd.DataFrame,    # cols: lmdb_split, lmdb_idx, n_h2o, n_co2, coop_pred
    target_path: Path = PREP / "cooperative_target.parquet",
) -> Dict[str, float]:
    """Pearson r between predicted cooperative term and B3.1-derived target.

    Uses only mixed-condition records where target is non-NaN.
    """
    target = pd.read_parquet(target_path)
    target = target.dropna(subset=["cooperative_target"])
    target = target[target["gas_mode"] == "mixed"]

    df = target.merge(
        predicted_cooperative,
        on=["lmdb_split", "lmdb_idx", "n_h2o", "n_co2"], how="inner")
    if len(df) < 5:
        return dict(coop_pearson=float("nan"), coop_n=len(df))
    pear = float(pearsonr(df["coop_pred"].values,
                          df["cooperative_target"].values)[0])
    spear = float(spearmanr(df["coop_pred"].values,
                            df["cooperative_target"].values)[0])
    return dict(coop_pearson=pear, coop_spearman=spear, coop_n=len(df))


# ─────────────────────────────────────────────────────────────────────
# 4. Counterfactual recovery (V7 specific, V8 informational)
# ─────────────────────────────────────────────────────────────────────
def counterfactual_recovery(
    do_intervention_results: pd.DataFrame,  # cols: lmdb_split, lmdb_idx, n_h2o, n_co2,
                                            #        delta_remove_cluster4
    d3_path: Path = REPO / "mechanism_analysis/outputs/D_counterfactuals/d3_cluster4_removal_per_mof.parquet",
) -> Dict[str, float]:
    """Pearson r between model's do(remove cluster 4) and D3 ground truth."""
    d3 = pd.read_parquet(d3_path)
    df = d3[["lmdb_split", "lmdb_idx", "n_h2o", "n_co2", "cluster4_contrib_total"]].merge(
        do_intervention_results, on=["lmdb_split", "lmdb_idx", "n_h2o", "n_co2"],
        how="inner")
    df = df.dropna(subset=["cluster4_contrib_total", "delta_remove_cluster4"])
    if len(df) < 5:
        return dict(cf_pearson=float("nan"), cf_n=len(df))
    pear = float(pearsonr(df["delta_remove_cluster4"].values,
                          df["cluster4_contrib_total"].values)[0])
    return dict(cf_pearson=pear, cf_n=len(df))


# ─────────────────────────────────────────────────────────────────────
# 5. Atom-type accuracy  (V8-style, sanity check)
# ─────────────────────────────────────────────────────────────────────
def atom_type_accuracy(
    predicted_logits: np.ndarray,       # (N_atoms, n_classes)
    targets: np.ndarray,                # (N_atoms,) ground truth atomic_numbers
) -> Dict[str, float]:
    """Top-1 accuracy of atom_type prediction."""
    pred_labels = np.argmax(predicted_logits, axis=1)
    acc = float(np.mean(pred_labels == targets))
    return dict(atom_type_acc=acc, n=len(targets))


# ─────────────────────────────────────────────────────────────────────
# Aggregate report
# ─────────────────────────────────────────────────────────────────────
def aggregate_report(
    model_name: str,
    predictions: pd.DataFrame | None = None,
    trained_alpha: pd.DataFrame | None = None,
    predicted_cooperative: pd.DataFrame | None = None,
    do_intervention_results: pd.DataFrame | None = None,
) -> Dict:
    """Build a single dict with all available metrics."""
    out = {"model": model_name}
    if predictions is not None:
        out.update(mae_id_and_ood(predictions))
    if trained_alpha is not None:
        out.update(mask_alpha_iou(trained_alpha))
    if predicted_cooperative is not None:
        out.update(cooperative_recovery(predicted_cooperative))
    if do_intervention_results is not None:
        out.update(counterfactual_recovery(do_intervention_results))
    return out


def compare_models(reports: Iterable[Dict]) -> pd.DataFrame:
    """Produce a side-by-side comparison DataFrame."""
    return pd.DataFrame(list(reports)).set_index("model").T
