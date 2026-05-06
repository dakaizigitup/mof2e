"""
Build environment labels for IRM (V7 stage 2 / V8 stage 2).

Each (lmdb_split, lmdb_idx, n_h2o, n_co2) sample is assigned to an environment:
  env_0 = pure_CO2  (n_h2o = 0, n_co2 > 0)
  env_1 = pure_H2O  (n_co2 = 0, n_h2o > 0)
  env_2 = mixed     (n_h2o > 0 and n_co2 > 0)

Output: causal_model/prep_data/env_labels.parquet
"""
import pickle, lmdb
from pathlib import Path
import pandas as pd

REPO = Path("/home/dell/autodl-tmp/lorafair")
PREP = REPO / "fairchem/causal_model/prep_data"


def env_id(h, c):
    if h == 0 and c > 0: return 0
    if c == 0 and h > 0: return 1
    if h > 0 and c > 0:  return 2
    return -1


def main():
    rows = []
    for split, path in [
        ("train", REPO / "data/mof2e/train/mof_only_aggregate_yrelaxed.lmdb"),
        ("val",   REPO / "data/mof2e/val/mof_val.lmdb"),
    ]:
        env = lmdb.open(str(path), readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            for idx, (_, v) in enumerate(txn.cursor()):
                s = pickle.loads(v)._store
                for (h, c), e in s["y_relaxed"].items():
                    rows.append({
                        "lmdb_split": split,
                        "lmdb_idx": int(idx),
                        "lmdb_name": s["name"],
                        "n_h2o": int(h),
                        "n_co2": int(c),
                        "env_id": env_id(h, c),
                    })
        env.close()

    df = pd.DataFrame(rows)
    df.to_parquet(PREP / "env_labels.parquet", index=False)
    print(f"Saved: {PREP / 'env_labels.parquet'}")
    print(f"Rows: {len(df)}")
    print()
    print("Environment distribution:")
    env_names = {0: "pure_CO2", 1: "pure_H2O", 2: "mixed", -1: "invalid"}
    counts = df["env_id"].value_counts().sort_index()
    for eid, n in counts.items():
        print(f"  env_{eid} ({env_names[eid]}): {n}")


if __name__ == "__main__":
    main()
