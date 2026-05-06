"""
Build V7 cooperative target.

For each (entry, condition) record, compute the cooperative ground truth:

  cooperative_target = E_actual − E_pure_h2o(h) − E_pure_co2(c) + E_baseline

where E_pure_h2o(h) is the actual recorded energy when n_co2=0, and similarly
for E_pure_co2(c). E_baseline is the reference at (0,0) — but since (0,0) is
typically not in y_relaxed, we approximate by 0 (relative scale).

If pure conditions are not available for a given entry, mark as NaN.

Chemistry hard floor (model-independent):
  - For pure_CO2 condition (n_h2o=0), cooperative MUST = 0
  - For pure_H2O condition (n_co2=0), cooperative MUST = 0

Output: causal_model/prep_data/cooperative_target.parquet
"""
import pickle, lmdb
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/dell/autodl-tmp/lorafair")
PREP = REPO / "fairchem/causal_model/prep_data"


def gas_mode(h, c):
    if h == 0 and c > 0: return "pure_CO2"
    if c == 0 and h > 0: return "pure_H2O"
    if h > 0 and c > 0:  return "mixed"
    return "neither"


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
                y = s["y_relaxed"]   # dict[(n_h2o, n_co2)] -> energy
                # Find pure references
                pure_h2o = {h: e for (h, c), e in y.items() if c == 0 and h > 0}
                pure_co2 = {c: e for (h, c), e in y.items() if h == 0 and c > 0}

                for (h, c), e_actual in y.items():
                    mode = gas_mode(h, c)

                    if mode in ("pure_CO2", "pure_H2O"):
                        # Chemistry hard floor: cooperative must be 0
                        coop_target = 0.0
                        coop_source = "chemistry_hard_floor"
                        is_floor = True
                    elif mode == "mixed":
                        e_h = pure_h2o.get(h)  # actual pure_H2O energy at same h
                        e_c = pure_co2.get(c)
                        if e_h is None or e_c is None:
                            # Cannot decompose; mark as NaN
                            coop_target = np.nan
                            coop_source = "missing_pure_reference"
                            is_floor = False
                        else:
                            # cooperative = E_actual - E_pure_H2O(h) - E_pure_CO2(c)
                            # (assuming linear baseline; remaining is the cooperative term)
                            coop_target = float(e_actual - e_h - e_c)
                            coop_source = "actual_decomposition"
                            is_floor = False
                    else:
                        coop_target = np.nan
                        coop_source = "neither_gas"
                        is_floor = False

                    rows.append({
                        "lmdb_split": split,
                        "lmdb_idx": int(idx),
                        "lmdb_name": s["name"],
                        "n_h2o": int(h),
                        "n_co2": int(c),
                        "gas_mode": mode,
                        "energy_actual": float(e_actual),
                        "cooperative_target": coop_target,
                        "source": coop_source,
                        "is_chemistry_floor": is_floor,
                    })
        env.close()

    df = pd.DataFrame(rows)
    out = PREP / "cooperative_target.parquet"
    df.to_parquet(out, index=False)

    print(f"Saved: {out}")
    print(f"Rows: {len(df)}")
    print()
    print("Source breakdown:")
    print(df["source"].value_counts())
    print()
    print("Cooperative target stats by gas mode (non-NaN):")
    print(df.dropna(subset=["cooperative_target"]).groupby("gas_mode")[
        "cooperative_target"].describe().round(3))


if __name__ == "__main__":
    main()
