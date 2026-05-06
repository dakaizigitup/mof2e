"""
Phase 1 - Step 2: NOTEARS 因果发现
在 MOF 级别数据上运行 NOTEARS，自动发现变量间的因果 DAG。
然后与领域知识因果图对比，输出最终因果图。
"""
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

out_dir = "/home/dell/autodl-tmp/lorafair/causal_analysis"

# ==================== 1. 加载数据 ====================
df = pd.read_csv(f"{out_dir}/df_mof_level.csv")
with open(f"{out_dir}/feature_mappings.json") as f:
    mappings = json.load(f)

print(f"MOF 级别数据: {df.shape}")
print(f"列: {list(df.columns)}")

# ==================== 2. 选择 NOTEARS 的变量 ====================
# 用可解释名称
# 注意：NOTEARS 对编码后的分类变量效果不好，
# 我们用连续 + 二值变量为主，分类变量用 one-hot 的主成分

# 核心变量
notears_cols = [
    "LCD", "PLD", "ASA", "AV_VF", "AV_cm3g",  # 孔结构 (连续，已 z-score)
    "OMS",                                       # 二值
    "lig1_length",                                # 配体长度 (连续)
    "energy_mean",                                # 目标
]
df_nt = df[notears_cols].copy()

# 标准化配体长度
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_nt["lig1_length"] = scaler.fit_transform(df_nt[["lig1_length"]])

print(f"\nNOTEARS 输入: {df_nt.shape}")
print(df_nt.describe().round(3).to_string())

# ==================== 3. 运行 NOTEARS ====================
print("\n" + "=" * 60)
print("运行 NOTEARS 因果发现...")
print("=" * 60)

from causalnex.structure.notears import from_pandas

# tabu_child_nodes: energy_mean 不能是任何变量的原因（它是最终目标）
# 但 NOTEARS 的 API 用 tabu_edges 来禁止特定边
# 我们禁止 energy_mean -> 任何孔结构/OMS 的边（因果方向不对）
tabu_edges = []
for col in notears_cols:
    if col != "energy_mean":
        tabu_edges.append(("energy_mean", col))  # 禁止 energy → X

# 运行
sm = from_pandas(
    df_nt,
    tabu_edges=tabu_edges,
    w_threshold=0.1,       # 边权重阈值（越大越严格，保留更少的边）
    max_iter=200,
)

print(f"\n发现的边 (共 {len(sm.edges())} 条):")
for u, v in sorted(sm.edges(), key=lambda e: abs(sm[e[0]][e[1]]["weight"]), reverse=True):
    w = sm[u][v]["weight"]
    print(f"  {u:15s} → {v:15s}  weight={w:+.4f}")

# ==================== 4. 不同阈值的边 ====================
print("\n" + "=" * 60)
print("不同阈值下的边数量:")
print("=" * 60)
for threshold in [0.05, 0.1, 0.2, 0.3, 0.5]:
    sm_t = from_pandas(df_nt, tabu_edges=tabu_edges, w_threshold=threshold, max_iter=200)
    edges_to_energy = [(u, v) for u, v in sm_t.edges() if v == "energy_mean"]
    print(f"  threshold={threshold:.2f}: {len(sm_t.edges())} edges total, {len(edges_to_energy)} → energy_mean")
    for u, v in edges_to_energy:
        print(f"    {u} → energy_mean  w={sm_t[u][v]['weight']:+.4f}")

# ==================== 5. 相关性热力图（对比用）====================
print("\n绘制相关性热力图...")
corr = df_nt.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            square=True, linewidths=0.5)
ax.set_title("Feature Correlation Matrix (MOF-level, Train set)", fontsize=14)
plt.tight_layout()
fig.savefig(f"{out_dir}/correlation_heatmap.png", dpi=150)
print(f"  保存: {out_dir}/correlation_heatmap.png")

# ==================== 6. 因果图可视化 ====================
print("\n绘制 NOTEARS 因果图...")

# 用 networkx 可视化
import networkx as nx

G = nx.DiGraph()
for col in notears_cols:
    G.add_node(col)

for u, v in sm.edges():
    w = sm[u][v]["weight"]
    G.add_edge(u, v, weight=w)

# 节点颜色
node_colors = []
for n in G.nodes():
    if n == "energy_mean":
        node_colors.append("#FF6B6B")  # 红色 - 目标
    elif n == "OMS":
        node_colors.append("#4ECDC4")  # 青色 - 二值
    elif n in ["LCD", "PLD", "ASA", "AV_VF", "AV_cm3g"]:
        node_colors.append("#45B7D1")  # 蓝色 - 孔结构
    else:
        node_colors.append("#96CEB4")  # 绿色 - 配体

# 边颜色和宽度
edge_colors = []
edge_widths = []
for u, v in G.edges():
    w = G[u][v]["weight"]
    edge_colors.append("red" if w < 0 else "blue")
    edge_widths.append(min(abs(w) * 5, 4))

fig, ax = plt.subplots(figsize=(14, 10))
pos = nx.spring_layout(G, k=2, seed=42)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                       arrows=True, arrowsize=20, arrowstyle="-|>",
                       connectionstyle="arc3,rad=0.1", ax=ax)

# 边标签
edge_labels = {(u, v): f"{G[u][v]['weight']:+.3f}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)

ax.set_title("NOTEARS Causal DAG (w_threshold=0.1)\nBlue=positive, Red=negative", fontsize=14)
plt.tight_layout()
fig.savefig(f"{out_dir}/notears_dag.png", dpi=150)
print(f"  保存: {out_dir}/notears_dag.png")

# ==================== 7. 领域知识 DAG 对比 ====================
print("\n" + "=" * 60)
print("领域知识 DAG vs NOTEARS DAG 对比")
print("=" * 60)

# 领域知识预期的因果边
domain_edges = {
    # 化学性质 → 孔结构
    ("OMS", "LCD"): "OMS 影响金属配位 → 影响孔腔大小",
    ("OMS", "PLD"): "OMS 影响金属配位 → 影响最小孔径",
    ("lig1_length", "LCD"): "配体越长 → 孔越大",
    ("lig1_length", "PLD"): "配体越长 → 孔径越大",
    ("lig1_length", "ASA"): "配体越长 → 表面积越大",

    # 孔结构 → 吸附能
    ("PLD", "energy_mean"): "孔径决定分子能否进入 → 影响吸附",
    ("ASA", "energy_mean"): "表面积 → 吸附位点多少",
    ("AV_VF", "energy_mean"): "孔隙率 → 吸附容量上限",

    # 化学性质 → 吸附能（直接效应）
    ("OMS", "energy_mean"): "OMS 直接提供强吸附位点",
    ("lig1_length", "energy_mean"): "配体化学性质影响吸附（间接）",
}

print("\n预期边        NOTEARS发现?  NOTEARS权重")
print("-" * 60)
for (u, v), reason in domain_edges.items():
    if G.has_edge(u, v):
        w = G[u][v]["weight"]
        status = f"✅ 发现  w={w:+.4f}"
    elif G.has_edge(v, u):
        w = G[v][u]["weight"]
        status = f"⚠️ 反向  w={w:+.4f}"
    else:
        status = "❌ 未发现"
    print(f"  {u:15s} → {v:15s}  {status}")
    print(f"    物理解释: {reason}")

# 意外发现的边（NOTEARS 发现但领域知识未预期）
notears_edges = set(G.edges())
domain_edge_set = set(domain_edges.keys())
unexpected = notears_edges - domain_edge_set
if unexpected:
    print(f"\nNOTEARS 发现但领域知识未预期的边 ({len(unexpected)} 条):")
    for u, v in unexpected:
        w = G[u][v]["weight"]
        print(f"  {u:15s} → {v:15s}  w={w:+.4f}")

# ==================== 8. 保存结果 ====================
results = {
    "notears_edges": [{"from": u, "to": v, "weight": float(sm[u][v]["weight"])}
                      for u, v in sm.edges()],
    "domain_comparison": {}
}
for (u, v), reason in domain_edges.items():
    found = G.has_edge(u, v)
    reversed_found = G.has_edge(v, u)
    results["domain_comparison"][f"{u}->{v}"] = {
        "reason": reason,
        "found": found,
        "reversed": reversed_found,
        "weight": float(G[u][v]["weight"]) if found else (float(G[v][u]["weight"]) if reversed_found else None),
    }

with open(f"{out_dir}/notears_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n结果已保存: {out_dir}/notears_results.json")
print("完成！")
