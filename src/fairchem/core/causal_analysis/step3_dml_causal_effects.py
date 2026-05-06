"""
Phase 2: 因果效应估计 (Double Machine Learning)
用 DML 逐个估计每个全局特征对吸附能的因果效应（ATE/CATE），
控制混淆变量，区分真因果 vs 伪相关。

输出：
  - 每个特征的 ATE + 95% 置信区间
  - OMS 的 CATE（按金属类型分群）
  - 汇总排序表
"""
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

out_dir = "/home/dell/autodl-tmp/lorafair/causal_analysis"

# ==================== 1. 加载数据 ====================
df = pd.read_csv(f"{out_dir}/df_expanded.csv")
print(f"数据: {df.shape}, 列: {list(df.columns)}")

# 特征列
pore_features = ["LCD", "PLD", "LFPD", "ASA", "AV_VF", "AV_cm3g", "has_NASA"]
all_features = pore_features + ["OMS", "Metal_enc", "Topology_enc", "lig1_length", "n_h2o", "n_co2"]

# ==================== 2. DML 分析函数 ====================
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LassoCV

def estimate_ate_dml(df, treatment_col, outcome_col="energy", confounders=None,
                     treatment_type="continuous", n_splits=5):
    """
    用 LinearDML 估计 treatment_col 对 outcome_col 的 ATE。
    confounders: 混淆变量列名列表（控制这些变量后看 treatment 的因果效应）
    treatment_type: "continuous" 或 "binary"（二值也用回归器处理，避免 econml 的 classifier 限制）
    """
    Y = df[outcome_col].values
    T = df[treatment_col].values.astype(float).reshape(-1, 1)
    W = df[confounders].values if confounders else None

    # econml LinearDML 的 model_t 必须用回归器（即使 T 是二值的）
    model_t = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_y = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

    est = LinearDML(
        model_y=model_y,
        model_t=model_t,
        cv=min(n_splits, 5),
        random_state=42,
    )
    est.fit(Y=Y, T=T, W=W)

    ate = est.ate()
    ci = est.ate_interval(alpha=0.05)

    return {
        "ate": float(ate),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "significant": bool(ci[0] * ci[1] > 0),  # 置信区间不跨零
    }


# ==================== 3. 分析 1：OMS 的因果效应 ====================
print("\n" + "=" * 70)
print("分析 1：OMS 的因果效应（控制 Metal + 孔结构 + condition）")
print("=" * 70)

oms_confounders = ["Metal_enc", "LCD", "PLD", "ASA", "AV_VF", "AV_cm3g",
                   "lig1_length", "n_h2o", "n_co2"]
oms_result = estimate_ate_dml(df, "OMS", confounders=oms_confounders, treatment_type="binary")
print(f"  OMS ATE = {oms_result['ate']:.4f} eV")
print(f"  95% CI  = [{oms_result['ci_lower']:.4f}, {oms_result['ci_upper']:.4f}]")
print(f"  显著?    = {'是 ✅' if oms_result['significant'] else '否 ❌'}")
print(f"  解读: 控制金属类型和孔结构后，有 OMS 的 MOF 吸附能平均{'低' if oms_result['ate'] < 0 else '高'} {abs(oms_result['ate']):.3f} eV")

# ==================== 4. 分析 2：各孔结构变量的因果效应 ====================
print("\n" + "=" * 70)
print("分析 2：各孔结构变量的因果效应（控制化学组成 + condition）")
print("=" * 70)

results_all = {}
results_all["OMS"] = oms_result

# 对每个孔结构变量，控制 OMS + Metal + 其他孔结构 + condition
for feat in ["LCD", "PLD", "ASA", "AV_VF", "AV_cm3g", "lig1_length"]:
    # 混淆变量 = 除了自己以外的所有特征
    confounders = [c for c in all_features if c != feat and c != "energy"]
    try:
        result = estimate_ate_dml(df, feat, confounders=confounders)
        results_all[feat] = result
        sig = "✅" if result["significant"] else "❌"
        print(f"  {feat:15s}: ATE={result['ate']:+.4f}  CI=[{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]  {sig}")
    except Exception as e:
        print(f"  {feat:15s}: 失败 - {e}")
        results_all[feat] = {"ate": 0, "ci_lower": 0, "ci_upper": 0, "significant": False, "error": str(e)}

# ==================== 5. 分析 3：n_h2o 和 n_co2 的因果效应 ====================
print("\n" + "=" * 70)
print("分析 3：条件变量 (n_h2o, n_co2) 的因果效应")
print("=" * 70)

for feat in ["n_h2o", "n_co2"]:
    confounders = [c for c in all_features if c != feat and c != "energy"]
    try:
        result = estimate_ate_dml(df, feat, confounders=confounders)
        results_all[feat] = result
        sig = "✅" if result["significant"] else "❌"
        print(f"  {feat:15s}: ATE={result['ate']:+.4f}  CI=[{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]  {sig}")
    except Exception as e:
        print(f"  {feat:15s}: 失败 - {e}")

# ==================== 6. OMS 的 CATE（按金属类型分群）====================
print("\n" + "=" * 70)
print("分析 4：OMS 的条件因果效应 CATE（按金属类型分群）")
print("=" * 70)

cate_by_metal = {}
for metal, group in df.groupby("Metal"):
    if len(group) < 50:  # 样本太少跳过
        continue
    # 检查 OMS 是否有变异
    if group["OMS"].nunique() < 2:
        cate_by_metal[metal] = {"ate": float("nan"), "note": "OMS no variation"}
        print(f"  {metal:10s} (n={len(group):4d}): OMS 无变异（全是 0 或全是 1），跳过")
        continue
    try:
        conf = ["LCD", "PLD", "ASA", "AV_VF", "AV_cm3g", "lig1_length", "n_h2o", "n_co2"]
        result = estimate_ate_dml(group, "OMS", confounders=conf, treatment_type="binary")
        cate_by_metal[metal] = result
        sig = "✅" if result["significant"] else "❌"
        print(f"  {metal:10s} (n={len(group):4d}): ATE={result['ate']:+.4f}  CI=[{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]  {sig}")
    except Exception as e:
        cate_by_metal[metal] = {"ate": float("nan"), "error": str(e)}
        print(f"  {metal:10s} (n={len(group):4d}): 失败 - {e}")

# ==================== 7. OMS 的 CATE（按 condition 分群）====================
print("\n" + "=" * 70)
print("分析 5：OMS 的条件因果效应 CATE（按吸附条件分群）")
print("=" * 70)

cate_by_cond = {}
for (h, c), group in df.groupby(["n_h2o", "n_co2"]):
    cond_label = f"h2o={h}_co2={c}"
    if group["OMS"].nunique() < 2:
        print(f"  {cond_label} (n={len(group)}): OMS 无变异，跳过")
        continue
    try:
        conf = ["Metal_enc", "LCD", "PLD", "ASA", "AV_VF", "AV_cm3g", "lig1_length"]
        result = estimate_ate_dml(group, "OMS", confounders=conf, treatment_type="binary")
        cate_by_cond[cond_label] = result
        sig = "✅" if result["significant"] else "❌"
        print(f"  {cond_label:15s} (n={len(group):4d}): ATE={result['ate']:+.4f}  CI=[{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]  {sig}")
    except Exception as e:
        print(f"  {cond_label:15s}: 失败 - {e}")

# ==================== 8. 汇总排序表 ====================
print("\n" + "=" * 70)
print("因果效应汇总排序（按 |ATE| 从大到小）")
print("=" * 70)

summary_rows = []
for feat, res in results_all.items():
    if "error" in res:
        continue
    summary_rows.append({
        "Feature": feat,
        "ATE": res["ate"],
        "|ATE|": abs(res["ate"]),
        "CI_lower": res["ci_lower"],
        "CI_upper": res["ci_upper"],
        "Significant": res["significant"],
    })

df_summary = pd.DataFrame(summary_rows).sort_values("|ATE|", ascending=False)
print(df_summary.to_string(index=False))

# ==================== 9. 可视化 ====================
print("\n绘制因果效应排序图...")

fig, ax = plt.subplots(figsize=(12, 7))
df_plot = df_summary.sort_values("ATE")
colors = ["#FF6B6B" if s else "#CCCCCC" for s in df_plot["Significant"]]

bars = ax.barh(df_plot["Feature"], df_plot["ATE"], color=colors, edgecolor="gray", height=0.6)

# 误差线
for i, (_, row) in enumerate(df_plot.iterrows()):
    ax.plot([row["CI_lower"], row["CI_upper"]], [i, i], color="black", linewidth=1.5)
    ax.plot([row["CI_lower"]], [i], marker="|", color="black", markersize=10)
    ax.plot([row["CI_upper"]], [i], marker="|", color="black", markersize=10)

ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Average Treatment Effect (ATE) on Energy (eV)", fontsize=12)
ax.set_title("Causal Effects of Global Features on Adsorption Energy\n(DML, controlling for confounders, 95% CI)\nRed = Significant, Gray = Not Significant",
             fontsize=13)
plt.tight_layout()
fig.savefig(f"{out_dir}/dml_ate_ranking.png", dpi=150)
print(f"  保存: {out_dir}/dml_ate_ranking.png")

# OMS CATE by condition 可视化
if cate_by_cond:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    cond_labels = list(cate_by_cond.keys())
    ates = [cate_by_cond[c]["ate"] for c in cond_labels]
    ci_lows = [cate_by_cond[c]["ci_lower"] for c in cond_labels]
    ci_highs = [cate_by_cond[c]["ci_upper"] for c in cond_labels]
    errors = [[a - l for a, l in zip(ates, ci_lows)],
              [h - a for a, h in zip(ates, ci_highs)]]

    ax2.bar(cond_labels, ates, yerr=errors, capsize=5, color="#4ECDC4", edgecolor="gray")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("OMS ATE on Energy (eV)", fontsize=12)
    ax2.set_title("OMS Causal Effect by Adsorption Condition\n(More negative = OMS helps more)", fontsize=13)
    plt.tight_layout()
    fig2.savefig(f"{out_dir}/dml_oms_cate_condition.png", dpi=150)
    print(f"  保存: {out_dir}/dml_oms_cate_condition.png")

# ==================== 10. 保存结果 ====================
all_results = {
    "ate_summary": {k: v for k, v in results_all.items()},
    "oms_cate_by_metal": cate_by_metal,
    "oms_cate_by_condition": cate_by_cond,
}
# 处理 nan for JSON
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

with open(f"{out_dir}/dml_results.json", "w") as f:
    json.dump(clean_for_json(all_results), f, indent=2, ensure_ascii=False)

print(f"\n结果已保存: {out_dir}/dml_results.json")
print("Phase 2 完成！")
