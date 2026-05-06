"""
Build OOD splits for V7/V8 evaluation.

Three OOD test sets (within train; original val also kept as standard OOD):
  - OOD-Topology: hold-out {lvt, fcu, rna, pts, kgd, sra}
  - OOD-Metal:    hold-out main metals in {La, Ce, Ag, Tb, Eu, Nd}
  - OOD-Combined: union

Output: causal_model/prep_data/ood_split.json
"""
import json, lmdb, pickle
from pathlib import Path
import pandas as pd
from pymatgen.core.periodic_table import Element

REPO = Path("/home/dell/autodl-tmp/lorafair")
PREP = REPO / "fairchem/causal_model/prep_data"
PREP.mkdir(parents=True, exist_ok=True)

OOD_TOPO = {"lvt", "fcu", "rna", "pts", "kgd", "sra"}
OOD_METALS = {"La", "Ce", "Ag", "Tb", "Eu", "Nd"}


def load_lmdb_meta(path, split_name):
    rows = []
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        for idx, (_, v) in enumerate(txn.cursor()):
            s = pickle.loads(v)._store
            zs = set(int(z) for z in s["atomic_numbers"].long().tolist())
            metals = sorted([Element.from_Z(z).symbol for z in zs if Element.from_Z(z).is_metal])
            rows.append({
                "lmdb_split": split_name,
                "lmdb_idx": idx,
                "lmdb_name": s["name"],
                "metals": metals,
                "main_metal": metals[0] if metals else None,
            })
    env.close()
    return rows


def main():
    csv = pd.read_csv(REPO / "data/causal_analysis_data.csv",
                     usecols=["lmdb_name", "Topologies(拓扑)"]).drop_duplicates("lmdb_name")
    csv = csv.rename(columns={"Topologies(拓扑)": "topology"})
    csv["topology"] = csv["topology"].fillna("UNKNOWN")
    topo_map = dict(zip(csv["lmdb_name"], csv["topology"]))

    train = pd.DataFrame(load_lmdb_meta(REPO / "data/mof2e/train/mof_only_aggregate_yrelaxed.lmdb", "train"))
    val = pd.DataFrame(load_lmdb_meta(REPO / "data/mof2e/val/mof_val.lmdb", "val"))

    train["topology"] = train["lmdb_name"].map(topo_map).fillna("UNKNOWN")
    val["topology"] = val["lmdb_name"].map(topo_map).fillna("UNKNOWN")

    # OOD masks (within train)
    def by_topo(df):
        return df["topology"].isin(OOD_TOPO)
    def by_metal(df):
        return df["main_metal"].isin(OOD_METALS)

    train_ood_topo = train[by_topo(train)]
    train_ood_metal = train[by_metal(train)]
    train_ood_combined = train[by_topo(train) | by_metal(train)]
    train_id = train[~(by_topo(train) | by_metal(train))]

    splits = {
        "description": "OOD splits within train (original val kept as standard OOD)",
        "config": {
            "ood_topologies": sorted(OOD_TOPO),
            "ood_metals": sorted(OOD_METALS),
        },
        "in_distribution_train": {
            "n_entries": len(train_id),
            "lmdb_idxs": sorted(train_id["lmdb_idx"].tolist()),
        },
        "ood_topology": {
            "n_entries": len(train_ood_topo),
            "lmdb_idxs": sorted(train_ood_topo["lmdb_idx"].tolist()),
            "names_sample": train_ood_topo["lmdb_name"].head(10).tolist(),
        },
        "ood_metal": {
            "n_entries": len(train_ood_metal),
            "lmdb_idxs": sorted(train_ood_metal["lmdb_idx"].tolist()),
            "names_sample": train_ood_metal["lmdb_name"].head(10).tolist(),
        },
        "ood_combined": {
            "n_entries": len(train_ood_combined),
            "lmdb_idxs": sorted(train_ood_combined["lmdb_idx"].tolist()),
        },
        "standard_val": {
            "n_entries": len(val),
            "lmdb_idxs": sorted(val["lmdb_idx"].tolist()),
        },
    }

    out = PREP / "ood_split.json"
    out.write_text(json.dumps(splits, indent=2, ensure_ascii=False))
    print(f"Saved: {out}")
    print()
    print("Summary:")
    print(f"  in_distribution_train: {len(train_id)}")
    print(f"  ood_topology:          {len(train_ood_topo)}")
    print(f"  ood_metal:             {len(train_ood_metal)}")
    print(f"  ood_combined:          {len(train_ood_combined)}")
    print(f"  standard_val:          {len(val)}")
    print()
    print("OOD-topology distribution:")
    print(train_ood_topo["topology"].value_counts().to_string())
    print()
    print("OOD-metal distribution:")
    print(train_ood_metal["main_metal"].value_counts().to_string())


if __name__ == "__main__":
    main()
