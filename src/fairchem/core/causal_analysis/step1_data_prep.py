"""
Phase 1 - Step 1: 因果分析数据准备
把之前已经合并好的 causal_analysis_data.csv 进一步清洗编码，输出两份数据：
  1. df_expanded.csv  —— 展开后每条 (MOF, condition) 一行（3714行），用于 DML
  2. df_mof_level.csv —— 按 MOF 聚合（576行），用于 NOTEARS
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ==================== 1. 读取已合并的 CSV ====================
df_all = pd.read_csv("/home/dell/autodl-tmp/lorafair/data/causal_analysis_data.csv")
# 只用 Train 集 & 有完整孔结构的行
df = df_all[(df_all["split"] == "train") & df_all["LCD(最大腔体直径)"].notna()].copy()
df = df.reset_index(drop=True)
print(f"[1] Train 有全局特征: {len(df)} 条样本, {df['mof_name'].nunique()} 个独立 MOF")

# ==================== 4. 特征工程 ====================

# --- 4.1 孔结构数值特征 (log1p + z-score) ---
pore_cols_raw = [
    "LCD(最大腔体直径)", "PLD(最小孔径)", "LFPD(最大自由球体直径)",
    "ASA_m2_cm3(比表面积)", "AV_VF(孔隙率)", "AV_cm3_g(可接触体积)",
]
pore_cols_clean = ["LCD", "PLD", "LFPD", "ASA", "AV_VF", "AV_cm3g"]

for raw, clean in zip(pore_cols_raw, pore_cols_clean):
    if clean == "AV_VF":
        df[clean] = df[raw]  # 孔隙率 0-1 之间，不做 log
    else:
        df[clean] = np.log1p(df[raw])

# NASA > 0 作为二值
df["has_NASA"] = (df["NASA_m2_cm3(不可接触表面积)"] > 0).astype(float)

pore_features = pore_cols_clean + ["has_NASA"]

# z-score 标准化
scaler = StandardScaler()
df[pore_cols_clean] = scaler.fit_transform(df[pore_cols_clean])

# --- 4.2 Has_OMS -> 二值 ---
df["OMS"] = (df["Has_OMS(是否有缺陷)"] == "Yes").astype(int)

# --- 4.3 金属类型 -> top-K + Other ---
metal_counts = df.groupby("mof_name")["All_Metals(金属种类)"].first().value_counts()
top_metals = metal_counts.head(8).index.tolist()  # 前 8 种保留
df["Metal"] = df["All_Metals(金属种类)"].apply(lambda x: x if x in top_metals else "Other")
metal_le = LabelEncoder()
df["Metal_enc"] = metal_le.fit_transform(df["Metal"])
print(f"[4.3] 金属类型: {metal_le.classes_}")

# --- 4.4 拓扑 -> top-K + Other/UNKNOWN ---
topo_counts = df.groupby("mof_name")["Topologies(拓扑)"].first().value_counts()
top_topos = topo_counts.head(10).index.tolist()
df["Topology"] = df["Topologies(拓扑)"].fillna("UNKNOWN").apply(
    lambda x: x if x in top_topos else "Other"
)
topo_le = LabelEncoder()
df["Topology_enc"] = topo_le.fit_transform(df["Topology"])

# --- 4.5 配体长度 (连续) ---
df["lig1_length"] = df["length"].fillna(df["length"].median())

# --- 4.6 condition 特征 (已经是数值) ---
# n_h2o, n_co2 直接使用

# ==================== 5. 选择最终特征列 ====================
# 用于因果分析的列
feature_cols = (
    pore_features          # 7 个孔结构
    + ["OMS"]              # 1 个二值
    + ["Metal_enc"]        # 1 个编码
    + ["Topology_enc"]     # 1 个编码
    + ["lig1_length"]      # 1 个连续
    + ["n_h2o", "n_co2"]   # 2 个条件
)
target_col = "energy"
id_col = "mof_name"

df_analysis = df[[id_col] + feature_cols + [target_col, "Metal", "Topology"]].copy()
print(f"\n[5] 最终分析 DataFrame:")
print(f"    形状: {df_analysis.shape}")
print(f"    特征列: {feature_cols}")
print(f"    缺失: {df_analysis[feature_cols + [target_col]].isnull().sum().sum()}")

# ==================== 6. MOF 级别聚合（用于 NOTEARS）====================
# NOTEARS 需要独立样本，同一个 MOF 的多个 condition 不是独立的
# 方案：对每个 MOF，取 energy 的 mean 作为"平均吸附能"
agg_dict = {col: "first" for col in pore_features + ["OMS", "Metal_enc", "Topology_enc", "lig1_length", "Metal", "Topology"]}
agg_dict["energy"] = "mean"
agg_dict["n_h2o"] = "mean"  # 保留 condition 的均值（所有 MOF 基本相同）

df_mof = df_analysis.groupby("mof_name").agg(agg_dict).reset_index()
# 重命名
df_mof = df_mof.rename(columns={"energy": "energy_mean"})
print(f"\n[6] MOF 级别 DataFrame: {df_mof.shape}")

# ==================== 7. 保存 ====================
out_dir = "/home/dell/autodl-tmp/lorafair/causal_analysis"
df_analysis.to_csv(f"{out_dir}/df_expanded.csv", index=False)
df_mof.to_csv(f"{out_dir}/df_mof_level.csv", index=False)

# 保存编码器映射（方便后续解读）
import json
mappings = {
    "Metal": {str(k): str(v) for k, v in zip(metal_le.classes_, metal_le.transform(metal_le.classes_))},
    "Topology": {str(k): str(v) for k, v in zip(topo_le.classes_, topo_le.transform(topo_le.classes_))},
    "pore_scaler_mean": dict(zip(pore_cols_clean, scaler.mean_.tolist())),
    "pore_scaler_std": dict(zip(pore_cols_clean, scaler.scale_.tolist())),
}
with open(f"{out_dir}/feature_mappings.json", "w") as f:
    json.dump(mappings, f, indent=2, ensure_ascii=False)

print(f"\n[7] 已保存:")
print(f"    {out_dir}/df_expanded.csv ({len(df_analysis)} rows)")
print(f"    {out_dir}/df_mof_level.csv ({len(df_mof)} rows)")
print(f"    {out_dir}/feature_mappings.json")

# ==================== 8. 数据概览 ====================
print("\n" + "=" * 60)
print("数据概览")
print("=" * 60)
print(df_analysis[feature_cols + [target_col]].describe().round(3).to_string())

print(f"\n金属分布:")
print(df_analysis["Metal"].value_counts().to_string())

print(f"\n拓扑分布:")
print(df_analysis["Topology"].value_counts().to_string())

print(f"\nOMS 分布:")
print(df_analysis["OMS"].value_counts().to_string())
