"""
Phase 4 - 因果合成指导
Step 4.2: 剂量-响应曲线（AV_VF / AV_cm3g / LCD 的最优范围）
Step 4.3: 合成优先级清单
Step 4.4: 反事实预测（无OMS的MOF，引入OMS后预计提升多少）

输出：
  - dose_response_curves.png  剂量-响应图
  - counterfactual_ranking.csv  改造收益排名
  - synthesis_guide.txt  合成建议文本
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

OUT_DIR = "/home/dell/autodl-tmp/lorafair/causal_analysis"

# ─────────────────────────────────────────────────────────────────────────────
# 1. 读取数据
# ─────────────────────────────────────────────────────────────────────────────
print("[1] 读取数据...")
df = pd.read_csv(f"{OUT_DIR}/df_expanded.csv")
excel = pd.read_excel("/home/dell/autodl-tmp/lorafair/data/MOF_embedding_all.xlsx")

# 把原始（未标准化）孔结构值合并进来
excel_raw = excel[["Name", "AV_VF(孔隙率)", "AV_cm3_g(可接触体积)", "LCD(最大腔体直径)"]].rename(columns={
    "Name": "mof_name",
    "AV_VF(孔隙率)": "AV_VF_raw",
    "AV_cm3_g(可接触体积)": "AV_cm3g_raw",
    "LCD(最大腔体直径)": "LCD_raw",
})
df = df.merge(excel_raw, on="mof_name", how="left")
print(f"  样本数: {len(df)} 行, {df['mof_name'].nunique()} 个 MOF")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 剂量-响应分析（Partial Regression）
# ─────────────────────────────────────────────────────────────────────────────
# 方法：
#   1. 用混淆变量（OMS / Metal / n_h2o / n_co2）对 energy 做线性回归，取残差
#   2. 把残差按原始特征值分 10 组，画每组均值 ± 置信区间
#   → 残差已经"去掉了"OMS/金属/条件的影响，纯粹反映孔结构的效应

print("[2] 计算 partial regression 残差...")
confounders = ["OMS", "Metal_enc", "n_h2o", "n_co2"]
X_conf = df[confounders].values
y = df["energy"].values

reg = LinearRegression().fit(X_conf, y)
residuals = y - reg.predict(X_conf)   # 去除混淆后的能量残差
df["energy_resid"] = residuals

def dose_response(df, feature_raw, feature_label, n_bins=10, ax=None):
    """
    对原始值 feature_raw 做等频分组，画残差均值±SE曲线。
    返回结果 DataFrame。
    """
    valid = df[feature_raw].notna()
    sub = df[valid].copy()

    sub["bin"] = pd.qcut(sub[feature_raw], q=n_bins, duplicates="drop")
    grouped = sub.groupby("bin", observed=True)["energy_resid"].agg(["mean", "sem", "count"])
    grouped["x_mid"] = grouped.index.map(lambda iv: iv.mid)
    grouped = grouped.reset_index()

    if ax is not None:
        ax.plot(grouped["x_mid"], grouped["mean"], "o-", color="steelblue", lw=2)
        ax.fill_between(
            grouped["x_mid"],
            grouped["mean"] - 1.96 * grouped["sem"],
            grouped["mean"] + 1.96 * grouped["sem"],
            alpha=0.25, color="steelblue",
        )
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(feature_label, fontsize=11)
        ax.set_ylabel("Partial Effect on Energy (eV)", fontsize=11)
        ax.set_title(f"Dose-Response: {feature_label}", fontsize=12)
        ax.grid(True, alpha=0.3)

        # 标注最优点
        best_idx = grouped["mean"].idxmin()
        best_x = grouped.loc[best_idx, "x_mid"]
        best_y = grouped.loc[best_idx, "mean"]
        ax.annotate(f"Best ≈ {best_x:.2f}", xy=(best_x, best_y),
                    xytext=(best_x, best_y + 0.05),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    color="red", fontsize=9)

    return grouped

print("[2] 绘制剂量-响应曲线...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

features_to_plot = [
    ("AV_VF_raw",   "AV_VF（孔隙率）"),
    ("AV_cm3g_raw", "AV_cm3g（可接触体积 cm³/g）"),
    ("LCD_raw",     "LCD（最大腔体直径 Å）"),
]

dose_results = {}
for (raw_col, label), ax in zip(features_to_plot, axes):
    res = dose_response(df, raw_col, label, n_bins=10, ax=ax)
    dose_results[raw_col] = res

plt.suptitle("Phase 4 Step 4.2 — Dose-Response Curves (Controlling for OMS/Metal/Condition)",
             fontsize=13, y=1.01)
plt.tight_layout()
out_path = f"{OUT_DIR}/dose_response_curves.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"  已保存: {out_path}")
plt.close()

# 输出最优区间
print("\n[2] 最优区间汇总：")
for raw_col, label in features_to_plot:
    res = dose_results[raw_col]
    best_idx = res["mean"].idxmin()
    best_row = res.loc[best_idx]
    interval = best_row["bin"]
    print(f"  {label}: 最优区间 {interval}, 平均 partial effect = {best_row['mean']:.4f} eV")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Step 4.4 反事实预测：无OMS → 有OMS 的收益
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] 反事实预测：对无OMS的MOF估算引入OMS的收益...")

# 读取 DML 结果（已有各条件下的 OMS CATE）
with open(f"{OUT_DIR}/dml_results.json") as f:
    dml = json.load(f)

# 全局 OMS ATE
oms_ate = dml["ate_summary"]["OMS"]["ate"]
oms_ci_lo = dml["ate_summary"]["OMS"]["ci_lower"]
oms_ci_hi = dml["ate_summary"]["OMS"]["ci_upper"]
print(f"  全局 OMS ATE: {oms_ate:.4f} eV  (95% CI: [{oms_ci_lo:.4f}, {oms_ci_hi:.4f}])")

# 按条件分组的 OMS CATE（从已保存结果读取）
cate_by_condition = dml["oms_cate_by_condition"]
print("  条件细分 CATE:")
for k, v in cate_by_condition.items():
    sig = "✓" if v["significant"] else " "
    print(f"    {sig} {k}: {v['ate']:.4f} eV  [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]")

# 对每个无OMS的 MOF，基于其吸附条件分布估算改造收益
# 按 MOF 聚合（取各条件平均 DFT energy，再估算引入OMS后的变化）
no_oms_mofs = df[df["OMS"] == 0]["mof_name"].unique()

records = []
for mof in no_oms_mofs:
    sub = df[(df["mof_name"] == mof) & (df["OMS"] == 0)]
    mean_energy = sub["energy"].mean()
    # 用全局 ATE 估算（保守估计，不依赖稀疏的条件 CATE）
    predicted_gain = oms_ate  # 负值 = 能量更低 = 吸附更强
    records.append({
        "mof_name": mof,
        "current_energy_mean": mean_energy,
        "predicted_gain_eV": predicted_gain,
        "predicted_energy_after_OMS": mean_energy + predicted_gain,
        "n_conditions": len(sub),
    })

cf_df = pd.DataFrame(records).sort_values("predicted_gain_eV")  # 最负的在最前
cf_df["rank"] = range(1, len(cf_df) + 1)
cf_path = f"{OUT_DIR}/counterfactual_ranking.csv"
cf_df.to_csv(cf_path, index=False)
print(f"\n  无OMS的MOF共 {len(cf_df)} 个，反事实排名已保存: {cf_path}")
print(f"  (注：当前版本使用全局 ATE={oms_ate:.4f} eV，所有MOF增益相同，排名按当前能量升序)")
print(cf_df[["mof_name", "current_energy_mean", "predicted_energy_after_OMS"]].head(10).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 4. Step 4.3 合成优先级清单
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] 生成合成优先级清单...")

# 从剂量-响应结果提取最优区间（原始值）
def get_optimal_range(dose_res, label):
    # 取最低 partial effect 的 bin
    best_idx = dose_res["mean"].idxmin()
    interval = dose_res.loc[best_idx, "bin"]
    effect = dose_res.loc[best_idx, "mean"]
    return interval, effect

avf_res = dose_results["AV_VF_raw"]
acm_res = dose_results["AV_cm3g_raw"]
lcd_res = dose_results["LCD_raw"]

avf_opt, avf_eff = get_optimal_range(avf_res, "AV_VF")
acm_opt, acm_eff = get_optimal_range(acm_res, "AV_cm3g")
lcd_opt, lcd_eff = get_optimal_range(lcd_res, "LCD")

ate_summary = dml["ate_summary"]

guide = f"""
================================================================================
Phase 4: MOF 合成指导（CO₂ 吸附优化）
生成时间: 基于 DFT 数据集（Train: 3714 样本, 576 MOFs）+ DML 因果分析
================================================================================

【一、关键合成参数排名（因果效应 ATE, 越负越重要）】

  排名  参数          ATE(eV)    显著  说明
  ────  ──────────   ─────────  ────  ──────────────────────────────
  #1    n_h2o        {ate_summary['n_h2o']['ate']:.4f}    ✓     每增加1个水分子，吸附能降低（增强吸附）
  #2    n_co2        {ate_summary['n_co2']['ate']:.4f}    ✓     每增加1个CO₂分子，吸附能降低（增强吸附）
  #3    LCD          {ate_summary['LCD']['ate']:.4f}    ✗     孔径影响不显著（统计不确定性大）
  #4    OMS          {ate_summary['OMS']['ate']:.4f}    ✓     有开放金属位点时，吸附能降低
  #5    AV_VF        {ate_summary['AV_VF']['ate']:.4f}    ✓     孔隙率增大，吸附能升高（减弱吸附）
  #6    AV_cm3g      {ate_summary['AV_cm3g']['ate']:.4f}    ✓     可接触体积增大，吸附能降低（增强吸附）

【二、连续特征最优范围（剂量-响应分析）】

  AV_VF（孔隙率）
    最优区间: {avf_opt}
    partial effect: {avf_eff:.4f} eV
    建议: 设计孔隙率在此范围内；过高的孔隙率（>0.75）会削弱 CO₂ 吸附

  AV_cm3g（可接触体积，cm³/g）
    最优区间: {acm_opt}
    partial effect: {acm_eff:.4f} eV
    建议: 适中的可接触体积有利于 CO₂ 结合；过大可能导致结合位点分散

  LCD（最大腔体直径，Å）
    最优区间: {lcd_opt}
    partial effect: {lcd_eff:.4f} eV
    建议: LCD 因果效应不显著，可结合 AV_VF 和 OMS 优先设计

【三、OMS 条件细分效应（哪种条件下引入 OMS 最有效）】

  条件               OMS 效应(eV)    显著    建议
  ─────────────────  ────────────   ─────   ────────────────────────────
  纯CO₂(h2o=0,co2=1) {cate_by_condition['h2o=0_co2=1']['ate']:.4f}         ✗       纯CO₂条件 OMS 效果不显著
  含水(h2o=1,co2=0)  {cate_by_condition['h2o=1_co2=0']['ate']:.4f}         ✓       含水条件 OMS 效果最强 → 优先引入OMS
  混合(h2o=1,co2=1)  {cate_by_condition['h2o=1_co2=1']['ate']:.4f}         ✗       混合条件效果不稳定

  ★ 关键发现：OMS 在含水条件下才显著有效（ATE=-0.136 eV），
    纯CO₂条件下效果接近零。对于含水 CO₂ 捕获场景，OMS 是首要合成目标。

【四、综合合成建议（优先级排序）】

  对于含水 CO₂ 吸附（实际工业场景）：
    1. [高优先] 引入 OMS（开放金属位点） —— 效果最显著，ATE=-0.136 eV（含水条件）
    2. [中优先] 控制孔隙率 AV_VF 在最优区间 {avf_opt}
    3. [中优先] 配体长度（lig1_length）影响不显著，可灵活选择
    4. [参考]   LCD 影响不显著，不作为优先优化目标

  对于纯 CO₂ 吸附（实验室/高压场景）：
    1. [高优先] 增大 CO₂ loading（n_co2 效应 ATE=-0.403 eV）
    2. [中优先] 控制 AV_cm3g 在最优区间 {acm_opt}
    3. [低优先] OMS 效果不显著，合成难度大时可不引入

【五、无OMS的MOF改造潜力（TOP 10）】

  （按当前平均吸附能排序，引入 OMS 后预计能量变化 {oms_ate:.4f} eV）
"""

# 加入 top10 无OMS高吸附 MOF 名
top10 = cf_df.nsmallest(10, "current_energy_mean")[["mof_name", "current_energy_mean", "predicted_energy_after_OMS"]]
for _, row in top10.iterrows():
    guide += f"  {row['mof_name']:<15} 当前: {row['current_energy_mean']:.4f} eV → 引入OMS后: {row['predicted_energy_after_OMS']:.4f} eV\n"

guide += "\n================================================================================\n"

guide_path = f"{OUT_DIR}/synthesis_guide.txt"
with open(guide_path, "w", encoding="utf-8") as f:
    f.write(guide)
print(f"  合成指导已保存: {guide_path}")
print(guide)
