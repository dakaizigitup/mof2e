import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import pickle
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
import logging

warnings.filterwarnings('ignore')


class MultiLabelAttention(nn.Module):
    """简单注意力池化: 输入 [k, d] 输出 [d]"""
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim // 2, bias=True)
        self.v = nn.Linear(dim // 2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [k, d]
        if x.shape[0] == 1:
            return x.squeeze(0)
        scores = self.v(torch.tanh(self.W(x)))  # [k,1]
        attn = torch.softmax(scores.squeeze(-1), dim=0)  # [k]
        return (attn.unsqueeze(-1) * x).sum(dim=0)  # [d]




# new fusion
class GlobalConcatFuse(nn.Module):
    """
    将图级(global)特征降维后，广播到每个节点，与节点特征拼接，再线性映射回节点通道。
    out = node_features + alpha * Linear([node_features || global_per_node])
    """
    def __init__(
        self,
        node_channels: int = 128,
        global_dim: int = 128,
        proj_dim: int = 32,
        dropout_p: float = 0.1,
        init_alpha: float = 0.1
    ):
        super().__init__()
        self.node_channels = node_channels
        self.global_dim = global_dim
        self.proj_dim = proj_dim

        # 全局特征降维编码
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.LayerNorm(proj_dim)
        )

        # 拼接后的线性融合层 (128 + 32) -> 128
        self.fusion_linear = nn.Sequential(
            nn.Linear(node_channels + proj_dim, node_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.LayerNorm(node_channels)
        )

        # 可学习缩放系数，初始较小，减少对主干的影响
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, node_features: torch.Tensor, global_features: torch.Tensor, batch: torch.Tensor):
        """
        node_features: [N, C]  (例如 [189, 128])
        global_features: [B, G] (例如 [1, 128] 或 [batch_size, 128])
        batch: [N] 节点对应的图索引 (0..B-1)
        """
        # 1) 全局特征降维
        global_encoded = self.global_encoder(global_features)  # [B, proj_dim]

        # 2) 广播到每个节点
        # 注意：确保 global_features 的顺序与 batch 的图索引对应
        global_per_node = global_encoded[batch]                # [N, proj_dim]

        # 3) 拼接后线性融合
        fused_input = torch.cat([node_features, global_per_node], dim=-1)  # [N, C+proj_dim]
        fused_delta = self.fusion_linear(fused_input)                      # [N, C]

        # 4) 残差注入 + 缩放，限制 alpha 到 [0,1]
        a = torch.clamp(self.alpha, 0.0, 1.0)
        out = node_features + a * fused_delta
        return out




class MOFGlobalPropertyEncoder(nn.Module):
    """
    基于Excel数据的MOF全局属性编码器（方案B：多标签特征用embedding聚合）

    额外增强：
    - 统计每次 forward 中 name 的命中/缺失情况
    - name 不存在时返回 0 向量（避免 NaN 传播）
    """

    def reset_coverage_stats(self) -> None:
        self._coverage_total = 0
        self._coverage_found = 0
        self._coverage_missing = 0
        self._coverage_missing_names = set()

    def report_coverage_stats(self, prefix: str = "MOFGlobalPropertyEncoder") -> None:
        total = getattr(self, "_coverage_total", 0)
        found = getattr(self, "_coverage_found", 0)
        missing = getattr(self, "_coverage_missing", 0)
        if total <= 0:
            logging.info(f"[{prefix}] 覆盖率统计：本轮未统计到任何 name")
            return
        rate = 100.0 * found / max(total, 1)
        missing_names = sorted(list(getattr(self, "_coverage_missing_names", set())))
        show_n = 20
        sample = missing_names[:show_n]
        logging.info(
            f"[{prefix}] 覆盖率统计：found={found}/{total} ({rate:.2f}%), missing={missing}. "
            f"missing_sample(first {min(show_n, len(sample))})={sample}"
        )

    def __init__(
        self,
        excel_path: str,
        global_dim: int = 128,
        hidden_dim: int = 256,
        smiles_embedding_dim: int = 64,
        max_smiles_length: int = 512,
        device: str = 'cuda',
        multi_label_pool: str = 'mean',  # 'mean' | 'sum' | 'attn'
    ):
        super().__init__()
        self.excel_path = excel_path
        # 最终输出维度固定为 64（按“倒金字塔”方案）
        self.global_dim = 64
        self.hidden_dim = hidden_dim
        self.smiles_embedding_dim = smiles_embedding_dim
        self.max_smiles_length = max_smiles_length
        self.device = device
        self.multi_label_pool = multi_label_pool

        # 读表
        self.df = pd.read_excel(excel_path).fillna('')

        # 数值特征
        self.log1p_numeric_features = { # 对这些列先做 log1p 再标准化
            'LCD(最大腔体直径)', 'PLD(最小孔径)', 'LFPD(最大自由球体直径)',
            'ASA_m2_cm3(比表面积)', 'NASA_m2_cm3(不可接触表面积)',
            'AV_cm3_g(可接触体积)', 'NAV_cm3_g(不可接触体积)'
        }
        self.numeric_features = [
            'LCD(最大腔体直径)', 'PLD(最小孔径)', 'LFPD(最大自由球体直径)',
            'ASA_m2_cm3(比表面积)', 'NASA_m2_cm3(不可接触表面积)',
            'AV_VF(孔隙率)', 'AV_cm3_g(可接触体积)', 'NAV_cm3_g(不可接触体积)'
        ]

        # 分类特征（含多标签）
        self.categorical_features = [
            'All_Metals(金属种类)',       # 多标签
            'Has_OMS(是否有缺陷)',        # 单标签
            'Open_Metal_Sites(缺陷位点)', # 多标签
            'Topologies(拓扑)', 
            'Terminal Ligands(封端配体)'
        ]

        # 指明哪些是多标签
        self.multi_label_features = {
            'All_Metals(金属种类)',
            'Open_Metal_Sites(缺陷位点)'
        }

        # SMILES
        self.smiles_features = ['Metals(金属节点)', 'Ligand_1', 'Ligand_2', 'Ligand_3']

        # 配体附加特征（列名严格对齐 Excel：ligand1 用 length/structures/groups；ligand2 用 *.1；ligand3 用 *.2）
        self.ligand_features = {
            'Ligand_1': {'length': 'numeric', 'structures': 'categorical', 'groups': 'categorical'},
            'Ligand_2': {'length.1': 'numeric', 'structures.1': 'categorical', 'groups.1': 'categorical'},
            'Ligand_3': {'length.2': 'numeric', 'structures.2': 'categorical', 'groups.2': 'categorical'}
        }

        # 预处理
        self._preprocess_data()
        # 构建标准化器/标签编码器/词表
        self._build_encoders()
        # 构建网络
        self._build_neural_networks()
        # 注意力池（如果需要）
        if self.multi_label_pool == 'attn':
            # 给每个多标签特征独立 attention（也可合并使用一个）
            self.multi_label_attn = nn.ModuleDict()
            # 需要知道各多标签特征嵌入维度, 构建后在 _build_neural_networks 里保存 mapping
            for f, dim in self._multi_label_embed_dims.items():
                self.multi_label_attn[f] = MultiLabelAttention(dim)

        self.to(self.device)
        self.reset_coverage_stats()
        self._warned_enc_keys = set()  # 用于 enc-missing 警告去重

    @staticmethod
    def _normalize_mof_name(name: str) -> str:
        """统一 MOF 名称格式，用于建表和查找，确保匹配一致性。"""
        if not isinstance(name, str):
            name = str(name)
        # 1. 去除首尾空格 -> 2. 转大写 -> 3. 按下划线分割取第一部分
        return name.strip().upper().split('_')[0]

    def _preprocess_data(self):
        # 使用规范化后的 name 作为 key 建表
        self.name_to_idx = {
            self._normalize_mof_name(name): idx for idx, name in enumerate(self.df['Name'])
        }

    def _build_encoders(self):
        self.numeric_scalers = {}
        for f in self.numeric_features:
            if f in self.df.columns:
                scaler = StandardScaler()
                data = pd.to_numeric(self.df[f], errors='coerce').fillna(0)
                if f in self.log1p_numeric_features:
                    data = np.log1p(np.maximum(data.values, 0))
                    scaler.fit(data.reshape(-1, 1))
                else:
                    scaler.fit(data.values.reshape(-1, 1))
                self.numeric_scalers[f] = scaler

        self.categorical_encoders = {}
        self.categorical_dims = {}

        # 多标签：构建自定义 vocab
        # 单标签：用 LabelEncoder
        # 1) 先处理多标签
        # Metals
        if 'All_Metals(金属种类)' in self.df.columns:
            unique_metals = set()
            for entry in self.df['All_Metals(金属种类)']:
                if isinstance(entry, str) and entry.strip():
                    for m in entry.split(','):
                        if m.strip():
                            unique_metals.add(m.strip())
            self.metal_vocab = {m: i for i, m in enumerate(sorted(unique_metals))}
            self.categorical_dims['All_Metals(金属种类)'] = len(self.metal_vocab)
        else:
            self.metal_vocab = {}
            self.categorical_dims['All_Metals(金属种类)'] = 0

        # Open metal sites
        if 'Open_Metal_Sites(缺陷位点)' in self.df.columns:
            unique_sites = set()
            for entry in self.df['Open_Metal_Sites(缺陷位点)']:
                if isinstance(entry, str) and entry.strip():
                    for s in entry.split(','):
                        if s.strip():
                            unique_sites.add(s.strip())
            self.oms_vocab = {s: i for i, s in enumerate(sorted(unique_sites))}
            self.categorical_dims['Open_Metal_Sites(缺陷位点)'] = len(self.oms_vocab)
        else:
            self.oms_vocab = {}
            self.categorical_dims['Open_Metal_Sites(缺陷位点)'] = 0

        # 2) 其它单标签分类
        for f in self.categorical_features:
            if f in self.multi_label_features:
                continue  # 已处理
            if f in self.df.columns:
                enc = LabelEncoder()
                series = self.df[f].astype(str)
                enc.fit(series)
                self.categorical_encoders[f] = enc
                self.categorical_dims[f] = len(enc.classes_)
            else:
                self.categorical_encoders[f] = LabelEncoder().fit(['__dummy__'])
                self.categorical_dims[f] = 1

        # 配体分类（严格按 Excel 列名：structures / groups 及其 .1/.2 后缀）
        for ligand, attrs in self.ligand_features.items():
            for attr_name, attr_type in attrs.items():
                if attr_type == 'categorical':
                    col = attr_name
                    if col in self.df.columns:
                        enc = LabelEncoder()
                        series = self.df[col].astype(str)
                        enc.fit(series)
                        key = self._sanitize_key(f"{ligand}_{attr_name}")
                        self.categorical_encoders[key] = enc
                        self.categorical_dims[key] = len(enc.classes_)

        # [DEBUG] 打印配体分类 key 注册情况
        for ligand, attrs in self.ligand_features.items():
            for attr_name, attr_type in attrs.items():
                if attr_type == 'categorical':
                    key = self._sanitize_key(f"{ligand}_{attr_name}")
                    col = attr_name
                    col_ok = col in self.df.columns
                    enc_ok = key in self.categorical_encoders
                    dim_ok = key in self.categorical_dims
                    status = "OK" if (col_ok and enc_ok and dim_ok) else "MISSING"
                    logging.info(
                        f"[global_emb] _build_encoders | {ligand} | {attr_name} | "
                        f"col='{col}'({'✓' if col_ok else '✗'}) "
                        f"enc({'✓' if enc_ok else '✗'}) dim({'✓' if dim_ok else '✗'}) -> {status}"
                    )

        # 配体数值特征（length，严格按 Excel 列名 length/length.1/length.2）
        for ligand, attrs in self.ligand_features.items():
            for attr_name, attr_type in attrs.items():
                if attr_type == 'numeric':
                    col = attr_name
                    if col in self.df.columns:
                        mm = MinMaxScaler()
                        data = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
                        mm.fit(data.values.reshape(-1, 1))
                        self.numeric_scalers[f"{ligand}_{attr_name}"] = mm

        # SMILES vocab
        self._build_smiles_vocab()

    def _build_smiles_vocab(self):
        chars = set()
        for f in self.smiles_features:
            if f in self.df.columns:
                for smi in self.df[f]:
                    if isinstance(smi, str) and smi:
                        chars.update(list(smi))
        chars.update(['<PAD>', '<UNK>', '<START>', '<END>'])
        self.smiles_vocab = {ch: i for i, ch in enumerate(sorted(chars))}
        self.vocab_size = len(self.smiles_vocab)

    def _build_neural_networks(self):
        # 数值编码器（按“维度流”方案：8 -> 32 -> 64）
        numeric_dim = len(self.numeric_features)
        self.numeric_out_dim = 64
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, self.numeric_out_dim)
        )

        # 分类特征 embedding
        self.categorical_embeddings = nn.ModuleDict()
        total_cat_dim = 0
        self._multi_label_embed_dims = {}

        def calc_embed_dim(dim):
            return min(50, (dim + 1) // 2) if dim > 0 else 1

        # 主分类特征（含多标签）—— 正常计入 total_cat_dim
        for f in self.categorical_features:
            dim = self.categorical_dims.get(f, 0)
            embed_dim = calc_embed_dim(dim)
            emb_key = self._sanitize_key(f)
            self.categorical_embeddings[emb_key] = nn.Embedding(max(dim,1), embed_dim)
            total_cat_dim += embed_dim
            if f in self.multi_label_features:
                self._multi_label_embed_dims[f] = embed_dim

        # 配体分类 embedding：严格对齐 structures/groups 的 .1/.2 后缀（不计入 total_cat_dim）
        for ligand in ['Ligand_1','Ligand_2','Ligand_3']:
            if ligand == 'Ligand_1':
                cols = ['structures', 'groups']
            elif ligand == 'Ligand_2':
                cols = ['structures.1', 'groups.1']
            else:
                cols = ['structures.2', 'groups.2']

            for col in cols:
                key = self._sanitize_key(f"{ligand}_{col}")
                if key in self.categorical_dims:
                    dim = self.categorical_dims[key]
                    embed_dim = calc_embed_dim(dim)
                    self.categorical_embeddings[key] = nn.Embedding(max(dim, 1), embed_dim)
                    logging.info(
                        f"[global_emb] _build_neural_networks | Embedding created: {ligand}.{col} "
                        f"key='{key}' classes={dim} embed_dim={embed_dim}"
                    )
                    # 不再累计到 total_cat_dim，避免走全局分类通道
                else:
                    logging.warning(
                        f"[global_emb] _build_neural_networks | Embedding SKIPPED: {ligand}.{col} "
                        f"key='{key}' not in categorical_dims"
                    )

        # 分类通道输出维度：128（按方案）
        self.categorical_out_dim = 128
        self.categorical_encoder = nn.Sequential(
            nn.Linear(total_cat_dim, self.categorical_out_dim),
            nn.LayerNorm(self.categorical_out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.categorical_out_dim, self.categorical_out_dim)
        )

        # SMILES 编码
        self.smiles_embedding = nn.Embedding(self.vocab_size, self.smiles_embedding_dim)
        self.smiles_encoder = nn.LSTM(
            self.smiles_embedding_dim, # 输入维度
            self.smiles_embedding_dim,# 隐藏层维度
            batch_first=True, 
            bidirectional=True
        )
        self.smiles_processors = nn.ModuleDict()
        for f in self.smiles_features:
            key = self._sanitize_key(f)
            self.smiles_processors[key] = nn.Sequential(
                nn.Linear(self.smiles_embedding_dim * 2, self.smiles_embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

        # 配体特征融合（按“维度流”方案：输出 256）
        ligand_feature_dim = 0
        for ligand in ['Ligand_1','Ligand_2','Ligand_3']:
            ligand_feature_dim += self.smiles_embedding_dim  # SMILES
            ligand_feature_dim += 1  # length

            if ligand == 'Ligand_1':
                cols = ['structures', 'groups']
            elif ligand == 'Ligand_2':
                cols = ['structures.1', 'groups.1']
            else:
                cols = ['structures.2', 'groups.2']

            for col in cols:
                key = self._sanitize_key(f"{ligand}_{col}")
                if key in self.categorical_dims:
                    if key in self.categorical_embeddings:
                        dim = self.categorical_dims[key]
                        ligand_feature_dim += min(50, (dim + 1) // 2)

        self.ligand_out_dim = 256
        self.ligand_encoder = nn.Sequential(
            nn.Linear(ligand_feature_dim, self.ligand_out_dim),
            nn.LayerNorm(self.ligand_out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.ligand_out_dim, self.ligand_out_dim)
        )

        # 最终融合（倒金字塔压缩到 64 维输出）
        # 仅金属节点走全局 SMILES 通道；配体通道给予更大带宽
        final_input_dim = (
            self.numeric_out_dim +
            self.categorical_out_dim +
            self.smiles_embedding_dim +
            self.ligand_out_dim
        )

        self.final_encoder = nn.Sequential(
            nn.Linear(final_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.global_dim)
        )

    @staticmethod
    def _sanitize_key(name: str) -> str:
        return name.replace('(', '').replace(')', '').replace(' ', '_').replace('.', '_')

    def _encode_smiles(self, smiles: str) -> torch.Tensor:
        if not isinstance(smiles, str) or not smiles:
            return torch.zeros(self.max_smiles_length, dtype=torch.long)
        ids = []
        for ch in smiles[:self.max_smiles_length - 1]:
            ids.append(self.smiles_vocab.get(ch, self.smiles_vocab['<UNK>']))
        while len(ids) < self.max_smiles_length:
            ids.append(self.smiles_vocab['<PAD>'])
        return torch.tensor(ids, dtype=torch.long)

    # （方案B中不再使用 multi-hot，以下函数仅保留以防兼容）
    def _encode_metals(self, metals_str: str):
        raise NotImplementedError("方案B不再使用 multi-hot，请使用多标签嵌入聚合")

    def _encode_oms(self, oms_str: str):
        raise NotImplementedError("方案B不再使用 multi-hot，请使用多标签嵌入聚合")

    def _parse_multi_label_tokens(self, raw: str) -> List[str]:
        if not isinstance(raw, str) or not raw.strip():
            return []
        return [t.strip() for t in raw.split(',') if t.strip()]

    def _aggregate_multi_label_embeddings(self, feature_name: str, tokens: List[str]) -> torch.Tensor:
        """
        tokens: 已拆分好的字符串标签
        """
        if feature_name == 'All_Metals(金属种类)':
            vocab = self.metal_vocab
        elif feature_name == 'Open_Metal_Sites(缺陷位点)':
            vocab = self.oms_vocab
        else:
            vocab = {}

        emb_key = self._sanitize_key(feature_name)
        emb_layer = self.categorical_embeddings[emb_key]
        dim = emb_layer.embedding_dim

        valid_idx = []
        for t in tokens:
            if t in vocab:
                valid_idx.append(vocab[t])
        if len(valid_idx) == 0:
            return torch.zeros(dim, device=self.device)

        idx_tensor = torch.tensor(valid_idx, dtype=torch.long, device=self.device)  # [k]
        embs = emb_layer(idx_tensor)  # [k, dim]

        if self.multi_label_pool == 'mean':
            return embs.mean(dim=0)
        elif self.multi_label_pool == 'sum':
            return embs.sum(dim=0)
        elif self.multi_label_pool == 'attn':
            return self.multi_label_attn[feature_name](embs)
        else:
            raise ValueError(f"未知 multi_label_pool: {self.multi_label_pool}")

    def encode_mof_by_name(self, mof_name: str) -> torch.Tensor:
        if mof_name not in self.name_to_idx:
            return torch.zeros(self.global_dim, device=self.device)

        row = self.df.iloc[self.name_to_idx[mof_name]]
        feature_parts = []

        # 1. 数值特征 —— 改动点：全局仅使用 self.numeric_features（去掉配体 length）
        numeric_values = []
        for f in self.numeric_features:
            val = pd.to_numeric(row.get(f, 0), errors='coerce')
            if pd.isna(val):
                val = 0.0
            if f in self.log1p_numeric_features:
                val = float(np.log1p(max(float(val), 0.0)))
            scaler = self.numeric_scalers.get(f, None)
            if scaler is not None:
                try:
                    val = scaler.transform([[val]])[0][0]
                except Exception:
                    pass
            numeric_values.append(val)
        # 删除“配体数值”的全局叠加，length 只在“配体融合”用

        if numeric_values:
            num_tensor = torch.tensor(numeric_values, dtype=torch.float32, device=self.device)
            num_encoded = self.numeric_encoder(num_tensor.unsqueeze(0)).squeeze(0)
            feature_parts.append(num_encoded)

        # 2. 分类特征（含多标签）—— 改动点：不再把配体 structures/groups 放入 categorical_vecs
        categorical_vecs = []
        for f in self.categorical_features:
            if f in self.multi_label_features:
                tokens = self._parse_multi_label_tokens(row.get(f, ''))
                agg_vec = self._aggregate_multi_label_embeddings(f, tokens)
                categorical_vecs.append(agg_vec)
            else:
                enc = self.categorical_encoders.get(f, None)
                emb_key = self._sanitize_key(f)
                if emb_key not in self.categorical_embeddings:
                    embed_dim = min(50, (self.categorical_dims.get(f,1)+1)//2)
                    categorical_vecs.append(torch.zeros(embed_dim, device=self.device))
                    continue
                emb_layer = self.categorical_embeddings[emb_key]
                if enc is None:
                    categorical_vecs.append(torch.zeros(emb_layer.embedding_dim, device=self.device))
                else:
                    raw_val = str(row.get(f, ''))
                    try:
                        idx = enc.transform([raw_val])[0]
                    except Exception:
                        idx = 0
                    idx_tensor = torch.tensor(idx, dtype=torch.long, device=self.device)
                    categorical_vecs.append(emb_layer(idx_tensor))

        # 删除“配体分类”小节（原先把 structures/groups 加入全局分类通道）—— 改动点

        if categorical_vecs:
            cat_concat = torch.cat(categorical_vecs, dim=0)
            cat_encoded = self.categorical_encoder(cat_concat.unsqueeze(0)).squeeze(0)
            feature_parts.append(cat_encoded)

        # 3. SMILES 特征 —— 改动点：仅金属节点走全局 SMILES 通道
        smiles_parts = []
        metal_f = 'Metals(金属节点)'
        smi = str(row.get(metal_f, ''))
        key = self._sanitize_key(metal_f)
        if smi and smi != 'nan':
            ids = self._encode_smiles(smi).to(self.device)
            emb_seq = self.smiles_embedding(ids)
            lstm_out, (h, c) = self.smiles_encoder(emb_seq.unsqueeze(0))
            final_hidden = torch.cat([h[0], h[1]], dim=-1).squeeze(0)
            proc = self.smiles_processors[key](final_hidden)
            smiles_parts.append(proc)
        else:
            smiles_parts.append(torch.zeros(self.smiles_embedding_dim, device=self.device))
        feature_parts.append(torch.cat(smiles_parts, dim=0))

        # 4. 配体融合 —— 唯一使用处：配体 SMILES + length + structures/groups
        ligand_vecs = []
        for ligand in ['Ligand_1','Ligand_2','Ligand_3']:
            sub_parts = []
            # SMILES
            smi = str(row.get(ligand, ''))
            key_lig = self._sanitize_key(ligand)
            if smi and smi != 'nan':
                ids = self._encode_smiles(smi).to(self.device)
                emb_seq = self.smiles_embedding(ids)
                lstm_out, (h, c) = self.smiles_encoder(emb_seq.unsqueeze(0))
                final_hidden = torch.cat([h[0], h[1]], dim=-1).squeeze(0)
                proc = self.smiles_processors[key_lig](final_hidden)
                sub_parts.append(proc)
            else:
                sub_parts.append(torch.zeros(self.smiles_embedding_dim, device=self.device))
            # length（仅在此使用）
            if ligand == 'Ligand_1':
                len_col = 'length'
            elif ligand == 'Ligand_2':
                len_col = 'length.1'
            else:
                len_col = 'length.2'
            if len_col in self.df.columns:
                val = pd.to_numeric(row.get(len_col, 0), errors='coerce')
                if pd.isna(val): val = 0.0
                scaler = self.numeric_scalers.get(f"{ligand}_{len_col}", None)
                if scaler is not None:
                    try:
                        val = scaler.transform([[val]])[0][0]
                    except Exception:
                        pass
                sub_parts.append(torch.tensor([val], dtype=torch.float32, device=self.device))
            else:
                sub_parts.append(torch.tensor([0.0], dtype=torch.float32, device=self.device))
            # structures / groups（仅在此使用）
            for attr in ['structures','groups']:
                if ligand == 'Ligand_1':
                    col = attr
                elif ligand == 'Ligand_2':
                    col = f"{attr}.1"
                else:
                    col = f"{attr}.2"

                key2 = self._sanitize_key(f"{ligand}_{col}")
                if key2 in self.categorical_embeddings:
                    emb_layer = self.categorical_embeddings[key2]
                    raw_val = str(row.get(col, ''))
                    enc = self.categorical_encoders.get(key2, None)
                    if enc is None:
                        if key2 not in self._warned_enc_keys:
                            logging.warning(
                                f"[global_emb] encode_mof_by_name | encoder missing for key='{key2}' "
                                f"({ligand}.{col}), using zero embedding"
                            )
                            self._warned_enc_keys.add(key2)
                        sub_parts.append(torch.zeros(emb_layer.embedding_dim, device=self.device))
                    else:
                        try:
                            idx = enc.transform([raw_val])[0]
                        except Exception:
                            idx = 0
                        sub_parts.append(emb_layer(torch.tensor(idx, dtype=torch.long, device=self.device)))
                else:
                    if key2 not in self._warned_enc_keys:
                        logging.warning(
                            f"[global_emb] encode_mof_by_name | embedding missing for key='{key2}' "
                            f"({ligand}.{col}), feature skipped"
                        )
                        self._warned_enc_keys.add(key2)
            ligand_vecs.append(torch.cat(sub_parts, dim=0))
        if ligand_vecs:
            lig_concat = torch.cat(ligand_vecs, dim=0)
            lig_encoded = self.ligand_encoder(lig_concat.unsqueeze(0)).squeeze(0)
            feature_parts.append(lig_encoded)

        # 5. 融合
        if feature_parts:
            final = torch.cat(feature_parts, dim=0)
            out = self.final_encoder(final.unsqueeze(0)).squeeze(0)
            return out
        else:
            return torch.zeros(self.global_dim, device=self.device)
    def encode_from_raw_values(self, raw_values: np.ndarray, col_names: list[str]) -> torch.Tensor:
        """根据传入的原始列值（顺序与 col_names 对应）编码成 embedding。
        该接口用于 SHAP，在不依赖 self.df 行索引的情况下复用相同的编码逻辑。
        """
        # 构造一个临时 Series 以复用现有 get(...) 逻辑
        row = pd.Series(raw_values, index=col_names)
        feature_parts = []

        # 1. 数值特征 —— 改动点：全局仅使用 self.numeric_features（去掉配体 length）
        numeric_values = []
        for f in self.numeric_features:
            val = pd.to_numeric(row.get(f, 0), errors='coerce')
            if pd.isna(val):
                val = 0.0
            if f in self.log1p_numeric_features:
                val = float(np.log1p(max(float(val), 0.0)))
            scaler = self.numeric_scalers.get(f, None)
            if scaler is not None:
                try:
                    val = scaler.transform([[val]])[0][0]
                except Exception:
                    pass
            numeric_values.append(val)
        # 删除“配体数值”的全局叠加，length 只在“配体融合”用

        if numeric_values:
            num_tensor = torch.tensor(numeric_values, dtype=torch.float32, device=self.device)
            num_encoded = self.numeric_encoder(num_tensor.unsqueeze(0)).squeeze(0)
            feature_parts.append(num_encoded)

        # 2. 分类特征（含多标签）—— 改动点：不再把配体 structures/groups 放入 categorical_vecs
        categorical_vecs = []
        for f in self.categorical_features:
            if f in self.multi_label_features:
                tokens = self._parse_multi_label_tokens(row.get(f, ''))
                agg_vec = self._aggregate_multi_label_embeddings(f, tokens)
                categorical_vecs.append(agg_vec)
            else:
                enc = self.categorical_encoders.get(f, None)
                emb_key = self._sanitize_key(f)
                if emb_key not in self.categorical_embeddings:
                    embed_dim = min(50, (self.categorical_dims.get(f,1)+1)//2)
                    categorical_vecs.append(torch.zeros(embed_dim, device=self.device))
                    continue
                emb_layer = self.categorical_embeddings[emb_key]
                if enc is None:
                    categorical_vecs.append(torch.zeros(emb_layer.embedding_dim, device=self.device))
                else:
                    raw_val = str(row.get(f, ''))
                    try:
                        idx = enc.transform([raw_val])[0]
                    except Exception:
                        idx = 0
                    idx_tensor = torch.tensor(idx, dtype=torch.long, device=self.device)
                    categorical_vecs.append(emb_layer(idx_tensor))

        # 删除“配体分类”小节（原先把 structures/groups 加入全局分类通道）—— 改动点

        if categorical_vecs:
            cat_concat = torch.cat(categorical_vecs, dim=0)
            cat_encoded = self.categorical_encoder(cat_concat.unsqueeze(0)).squeeze(0)
            feature_parts.append(cat_encoded)

        # 3. SMILES 特征 —— 改动点：仅金属节点走全局 SMILES 通道
        smiles_parts = []
        metal_f = 'Metals(金属节点)'
        smi = str(row.get(metal_f, ''))
        key = self._sanitize_key(metal_f)
        if smi and smi != 'nan':
            ids = self._encode_smiles(smi).to(self.device)
            emb_seq = self.smiles_embedding(ids)
            lstm_out, (h, c) = self.smiles_encoder(emb_seq.unsqueeze(0))
            final_hidden = torch.cat([h[0], h[1]], dim=-1).squeeze(0)
            proc = self.smiles_processors[key](final_hidden)
            smiles_parts.append(proc)
        else:
            smiles_parts.append(torch.zeros(self.smiles_embedding_dim, device=self.device))
        feature_parts.append(torch.cat(smiles_parts, dim=0))

        # 4. 配体融合 —— 唯一使用处：配体 SMILES + length + structures/groups
        ligand_vecs = []
        for ligand in ['Ligand_1','Ligand_2','Ligand_3']:
            sub_parts = []
            # SMILES
            smi = str(row.get(ligand, ''))
            key_lig = self._sanitize_key(ligand)
            if smi and smi != 'nan':
                ids = self._encode_smiles(smi).to(self.device)
                emb_seq = self.smiles_embedding(ids)
                lstm_out, (h, c) = self.smiles_encoder(emb_seq.unsqueeze(0))
                final_hidden = torch.cat([h[0], h[1]], dim=-1).squeeze(0)
                proc = self.smiles_processors[key_lig](final_hidden)
                sub_parts.append(proc)
            else:
                sub_parts.append(torch.zeros(self.smiles_embedding_dim, device=self.device))
            # length（仅在此使用）
            if ligand == 'Ligand_1':
                len_col = 'length'
            elif ligand == 'Ligand_2':
                len_col = 'length.1'
            else:
                len_col = 'length.2'
            if len_col in self.df.columns:
                val = pd.to_numeric(row.get(len_col, 0), errors='coerce')
                if pd.isna(val): val = 0.0
                scaler = self.numeric_scalers.get(f"{ligand}_{len_col}", None)
                if scaler is not None:
                    try:
                        val = scaler.transform([[val]])[0][0]
                    except Exception:
                        pass
                sub_parts.append(torch.tensor([val], dtype=torch.float32, device=self.device))
            else:
                sub_parts.append(torch.tensor([0.0], dtype=torch.float32, device=self.device))
            # structures / groups（仅在此使用）
            for attr in ['structures', 'groups']:
                if ligand == 'Ligand_1':
                    col = attr
                elif ligand == 'Ligand_2':
                    col = f"{attr}.1"
                else:
                    col = f"{attr}.2"
                key2 = self._sanitize_key(f"{ligand}_{col}")
                if key2 in self.categorical_embeddings:
                    emb_layer = self.categorical_embeddings[key2]
                    raw_val = str(row.get(col, ''))
                    enc = self.categorical_encoders.get(key2, None)
                    if enc is None:
                        if key2 not in self._warned_enc_keys:
                            logging.warning(
                                f"[global_emb] encode_from_raw_values | encoder missing for key='{key2}' "
                                f"({ligand}.{col}), using zero embedding"
                            )
                            self._warned_enc_keys.add(key2)
                        sub_parts.append(torch.zeros(emb_layer.embedding_dim, device=self.device))
                    else:
                        try:
                            idx = enc.transform([raw_val])[0]
                        except Exception:
                            idx = 0
                        sub_parts.append(emb_layer(torch.tensor(idx, dtype=torch.long, device=self.device)))
                else:
                    if key2 not in self._warned_enc_keys:
                        logging.warning(
                            f"[global_emb] encode_from_raw_values | embedding missing for key='{key2}' "
                            f"({ligand}.{col}), feature skipped"
                        )
                        self._warned_enc_keys.add(key2)
            ligand_vecs.append(torch.cat(sub_parts, dim=0))
        if ligand_vecs:
            lig_concat = torch.cat(ligand_vecs, dim=0)
            lig_encoded = self.ligand_encoder(lig_concat.unsqueeze(0)).squeeze(0)
            feature_parts.append(lig_encoded)

        # 5. 融合
        if feature_parts:
            final = torch.cat(feature_parts, dim=0)
            out = self.final_encoder(final.unsqueeze(0)).squeeze(0)
            return out
        else:
            return torch.zeros(self.global_dim, device=self.device)


    def forward(self, mof_names: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(mof_names, str):
            mof_names = [mof_names]

        embs = []
        for n in mof_names:
            raw_name = str(n)
            name = self._normalize_mof_name(raw_name)
            # 覆盖率统计
            self._coverage_total = getattr(self, "_coverage_total", 0) + 1
            if name in self.name_to_idx:
                self._coverage_found = getattr(self, "_coverage_found", 0) + 1
                emb = self.encode_mof_by_name(name)
            else:
                self._coverage_missing = getattr(self, "_coverage_missing", 0) + 1
                if not hasattr(self, "_coverage_missing_names"):
                    self._coverage_missing_names = set()
                
                if name not in self._coverage_missing_names:
                     logging.warning(f"MOF Global Feature Missing: '{name}' not found in Excel. Utilizing zero padding.")
                     self._coverage_missing_names.add(name)
                
                emb = torch.zeros(self.global_dim, device=next(self.parameters()).device)
            # 防御 NaN/Inf (最终兜底)
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                logging.warning(f"NaN or Inf detected in global embedding for MOF: {n}. Clamping to zero.")
                emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            embs.append(emb)

        return torch.stack(embs, dim=0)

    def save_encoders(self, path: str):
        state = {
            'numeric_scalers': self.numeric_scalers,
            'categorical_encoders': self.categorical_encoders,
            'categorical_dims': self.categorical_dims,
            'smiles_vocab': self.smiles_vocab,
            'metal_vocab': self.metal_vocab,
            'oms_vocab': self.oms_vocab,
            'name_to_idx': self.name_to_idx,
            'multi_label_pool': self.multi_label_pool
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_encoders(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.numeric_scalers = state['numeric_scalers']
        self.categorical_encoders = state['categorical_encoders']
        self.categorical_dims = state['categorical_dims']
        self.smiles_vocab = state['smiles_vocab']
        self.metal_vocab = state['metal_vocab']
        self.oms_vocab = state['oms_vocab']
        self.name_to_idx = state['name_to_idx']
        self.multi_label_pool = state.get('multi_label_pool', 'mean')
        print(f"Encoders loaded from {path}")
