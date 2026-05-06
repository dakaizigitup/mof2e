"""
Build V7 atom-level mask prior_α (per atom).

Two sources fused at 0.5:0.5:
  Source 1 (model-derived): A4 cluster 4 (free carboxylate) + A2 OMS-rich clusters
  Source 2 (pretrain-derived): MHG motif boundary (V2 mof_smiles_cache)

Output: causal_model/prep_data/prior_alpha_per_atom.parquet
  Columns: lmdb_split, lmdb_idx, atom_idx, prior_a4, prior_mhg, prior_alpha (fused), source
"""
import json, pickle, lmdb
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/dell/autodl-tmp/lorafair")
PREP = REPO / "fairchem/causal_model/prep_data"
PREP.mkdir(parents=True, exist_ok=True)

A1 = REPO / "mechanism_analysis/outputs/A1_site_units"
A2 = REPO / "mechanism_analysis/outputs/A2_site_motifs"
A4 = REPO / "mechanism_analysis/outputs/A4_subgraph_motifs"

OMS_RICH_CLUSTERS = {18, 22, 21, 33, 30, 4, 10}  # from A2
FREE_CARBOXYLATE_CLUSTER = 4                       # from A4

# fused weights
W_A4 = 0.5
W_MHG = 0.5


def load_lmdb_natoms():
    out = {}
    for split, path in [
        ("train", REPO / "data/mof2e/train/mof_only_aggregate_yrelaxed.lmdb"),
        ("val",   REPO / "data/mof2e/val/mof_val.lmdb"),
    ]:
        env = lmdb.open(str(path), readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            for idx, (_, v) in enumerate(txn.cursor()):
                s = pickle.loads(v)._store
                out[(split, idx)] = (int(s["natoms"]), s["name"])
        env.close()
    return out


def build_prior_a4(natoms_lookup):
    """Source 1: A4 cluster 4 + A2 OMS-rich clusters → atom-level prior_α."""
    # Atom level prior init: 0.3 default
    rows = {}
    for (split, idx), (natoms, name) in natoms_lookup.items():
        for ai in range(natoms):
            rows[(split, idx, ai)] = 0.3

    # Bump A2 OMS-rich metal sites to 0.7
    sites = pd.read_parquet(A1 / "metal_sites_index.parquet")
    labels = pd.read_parquet(A2 / "site_motif_labels.parquet")
    sites = sites.merge(
        labels[["lmdb_split", "lmdb_idx", "atom_idx", "cluster_mcs100_ms5"]],
        on=["lmdb_split", "lmdb_idx", "atom_idx"], how="left")
    oms_sites = sites[sites["cluster_mcs100_ms5"].isin(OMS_RICH_CLUSTERS)]
    print(f"OMS-rich sites (A2 clusters {OMS_RICH_CLUSTERS}): {len(oms_sites)}")
    for _, r in oms_sites.iterrows():
        key = (r["lmdb_split"], int(r["lmdb_idx"]), int(r["atom_idx"]))
        if key in rows:
            rows[key] = max(rows[key], 0.7)

    # General metals: 0.5
    for _, r in sites.iterrows():
        key = (r["lmdb_split"], int(r["lmdb_idx"]), int(r["atom_idx"]))
        if key in rows and rows[key] < 0.5:
            rows[key] = 0.5

    # Bump A4 cluster 4 (free carboxylate) atoms to 1.0
    motifs = pd.read_parquet(A4 / "motifs_with_clusters.parquet")
    c4_motifs = motifs[motifs["cluster_recommended"] == FREE_CARBOXYLATE_CLUSTER]
    print(f"Free carboxylate motifs (A4 cluster {FREE_CARBOXYLATE_CLUSTER}): {len(c4_motifs)}")
    for _, r in c4_motifs.iterrows():
        atoms = json.loads(r["atom_idxs_json"])
        for ai in atoms:
            key = (r["lmdb_split"], int(r["lmdb_idx"]), int(ai))
            if key in rows:
                rows[key] = 1.0
    return rows


def build_prior_mhg(natoms_lookup):
    """Source 2: MHG motif boundary from V2 mof_smiles_cache.

    The MHG cache encodes motif-level structure per MOF. We use its motif
    membership as a soft signal: atoms inside any "binding motif" (carboxylate
    / pyridine / imidazole) get higher prior; aromatic spacers stay at 0.3.

    Note: full MHG parsing requires loading the precomputed cache. For Phase 1
    prep, we do a lightweight version by re-using A4 motif identification
    (which uses RDKit SMARTS) since A4 already extracted the same chemical
    motif categories that MHG would identify.
    """
    rows = {}
    for (split, idx), (natoms, name) in natoms_lookup.items():
        for ai in range(natoms):
            rows[(split, idx, ai)] = 0.3

    motifs = pd.read_parquet(A4 / "subgraph_motifs.parquet")
    motif_priors = {
        "carboxylate": 0.9,
        "n5_ring": 0.8,
        "n6_ring": 0.7,
        "aromatic_6ring": 0.4,
    }
    for _, r in motifs.iterrows():
        atoms = json.loads(r["atom_idxs_json"])
        prior = motif_priors.get(r["motif_type"], 0.5)
        for ai in atoms:
            key = (r["lmdb_split"], int(r["lmdb_idx"]), int(ai))
            if key in rows:
                rows[key] = max(rows[key], prior)
    return rows


def main():
    print("Loading LMDB natoms...")
    natoms_lookup = load_lmdb_natoms()
    print(f"Total entries: {len(natoms_lookup)}")
    total_atoms = sum(n for n, _ in natoms_lookup.values())
    print(f"Total atoms: {total_atoms}")

    print("\nBuilding source 1: A4 cluster 4 + A2 OMS-rich")
    prior_a4 = build_prior_a4(natoms_lookup)
    print("\nBuilding source 2: MHG motif boundary (via A4 motif categories)")
    prior_mhg = build_prior_mhg(natoms_lookup)

    print("\nFusing sources (0.5 + 0.5)...")
    rows = []
    for (split, idx, ai) in prior_a4.keys():
        a4 = prior_a4[(split, idx, ai)]
        mhg = prior_mhg.get((split, idx, ai), 0.3)
        fused = W_A4 * a4 + W_MHG * mhg
        rows.append({
            "lmdb_split": split, "lmdb_idx": int(idx), "atom_idx": int(ai),
            "prior_a4": float(a4), "prior_mhg": float(mhg),
            "prior_alpha": float(fused),
            "source": "fused" if fused > 0.5 else "default",
        })

    df = pd.DataFrame(rows)
    out = PREP / "prior_alpha_per_atom.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved: {out}")
    print(f"Rows: {len(df)}")
    print(f"\nPrior_α distribution:")
    print(df["prior_alpha"].describe().round(3))
    print(f"\nHigh-prior atoms (α > 0.7): {(df['prior_alpha'] > 0.7).sum()}")
    print(f"Medium-prior atoms (0.4 < α ≤ 0.7): {((df['prior_alpha'] > 0.4) & (df['prior_alpha'] <= 0.7)).sum()}")
    print(f"Low-prior atoms (α ≤ 0.4): {(df['prior_alpha'] <= 0.4).sum()}")


if __name__ == "__main__":
    main()
