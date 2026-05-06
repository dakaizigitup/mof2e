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
warnings.filterwarnings('ignore')
class EquivariantGlobalFusion(nn.Module):
    """
    等变的全局特征融合模块
    这个模块用于把“节点特征”和“全局特征”融合在一起，
    保持物理上旋转等变（Equivariance）的性质。
    支持三种融合方式：加法、乘法门控、注意力调制。
    """
    def __init__(
        self,
        sphere_channels: int,
        global_dim: int,
        fusion_type: str = "additive",  # "additive", "multiplicative", "attention"
    ):
        super().__init__()
        self.sphere_channels = sphere_channels # 节点特征维度（球谐分量通道）
        self.global_dim = global_dim # 节点特征维度（球谐分量通道）
        self.fusion_type = fusion_type # 融合方式
        # 三种融合策略的实现方式
        if fusion_type == "additive":
            # 简单的加法融合 - 最安全的等变方式
            # 简单线性层映射后相加
            self.global_to_node = nn.Linear(global_dim, sphere_channels)
            
        elif fusion_type == "multiplicative":
            # 乘法门控融合
            self.global_to_gate = nn.Sequential(
                nn.Linear(global_dim, sphere_channels),
                nn.Sigmoid()
            )
            
        elif fusion_type == "attention":
            # 注意力调制 (tanh 形式)
            self.global_to_attention = nn.Sequential(
                nn.Linear(global_dim, sphere_channels),
                nn.Tanh()
            )
        
    def forward(
        self, 
        node_features: torch.Tensor,  # [num_nodes, sphere_channels] 每个原子的特征
        global_features: torch.Tensor,  # [batch_size, global_dim]  全局embedding
        batch: torch.Tensor,  # [num_nodes] 批次索引 每个节点属于哪个批次的索引
    ) -> torch.Tensor:
        """
        融合节点特征和全局特征，保持等变性
        """
        # # 把每个样本的全局特征扩展到它的所有节点上
        global_per_node = global_features[batch]  # [num_nodes, global_dim]
        # 三种融合方式
        if self.fusion_type == "additive":
            # 加法融合 - 保持等变性
            global_contribution = self.global_to_node(global_per_node)
            return node_features + global_contribution
            
        elif self.fusion_type == "multiplicative":
            # 乘法门控 - 保持等变性
            gate = self.global_to_gate(global_per_node)
            return node_features * gate
            
        elif self.fusion_type == "attention":
            # 注意力调制 - 保持等变性
            attention_weights = self.global_to_attention(global_per_node)
            return node_features * (1 + attention_weights)
        
        else:
            return node_features


## 无defect的编码
# ============================================================
# ✅ Excel 表格驱动的 MOF 全局属性编码器
# ============================================================
class MOFGlobalPropertyEncoder(nn.Module):
    """
    基于Excel数据的MOF全局属性编码器
    根据MOF名称自动查找并编码全局属性
    这个类负责：
    1️⃣ 从 Excel 文件中读取 MOF 的各种属性；
    2️⃣ 对不同类型特征（数值 / 分类 / SMILES / 配体）做预处理；
    3️⃣ 把它们转为统一维度的向量 embedding；
    4️⃣ 输出最终的 global embedding。

    用于 EquiformerV2-LoRA 的全局特征输入。
    """
    def __init__(
        self,
        excel_path: str = '/data0/wfz/code/fairchem/lora_model/config_lora/MOF_embedding_train-check580.xlsx',
        global_dim: int = 128,  # 输出embedding维度
        hidden_dim: int = 256,  # 隐藏层维度
        smiles_embedding_dim: int = 64, # SMILES嵌入维度
        max_smiles_length: int = 512, # SMILES最大字符长度
        device: str = 'cuda',
        
    ):
        super().__init__()

        # ----------------------
        # 初始化基本参数
        # ----------------------
        self.excel_path = excel_path
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.smiles_embedding_dim = smiles_embedding_dim
        self.max_smiles_length = max_smiles_length
        self.device = device
        # self.cache_dir = cache_dir
        
        # # 创建缓存目录
        # os.makedirs(cache_dir, exist_ok=True)

        # 读取 Excel 数据
        self.df = pd.read_excel(excel_path)
        self._preprocess_data() # 处理缺失值，创建索引
        self._build_encoders() # 为各类特征建立编码器
        
        # 数值特征字段（连续变量）
        self.numeric_features = [
            'LCD(最大腔体直径)', 'PLD(最小孔径)', 'LFPD(最大自由球体直径)',
            'ASA_m2_cm3(比表面积)', 'NASA_m2_cm3(不可接触表面积)', 
            'AV_VF(孔隙率)', 'AV_cm3_g(可接触体积)', 'NAV_cm3_g(不可接触体积)'
        ]
        # 分类特征字段（离散 one-hot）
        self.categorical_features = [
            'All_Metals(金属种类)', 'Has_OMS(是否有缺陷)', 'Open_Metal_Sites(缺陷位点)',
            'Topologies(拓扑)', 'Terminal Ligands(封端配体)'
        ]
        # SMILES 字符串字段
        self.smiles_features = [
            'Metals(金属节点)', 'Ligand_1', 'Ligand_2', 'Ligand_3'
        ]
        
        # 每个配体的子特征结构（含长度、结构、基团等）
        self.ligand_features = {
            'Ligand_1': {'length': 'numeric', 'structures': 'categorical', 'groups': 'categorical'},
            'Ligand_2': {'length': 'numeric', 'structures': 'categorical', 'groups': 'categorical'},
            'Ligand_3': {'length': 'numeric', 'structures': 'categorical', 'groups': 'categorical'}
        }
        
        self._build_neural_networks()
    # ---------------------------------------------------------
    # 数据预处理与编码器构建
    # ---------------------------------------------------------
    def _preprocess_data(self):
        """预处理数据"""
        # 处理缺失值
        self.df = self.df.fillna('')
        
        # 创建名称到索引的映射
        self.name_to_idx = {name: idx for idx, name in enumerate(self.df['Name'])}
        
        print(f"Loaded {len(self.df)} MOF structures")
        print(f"Available columns: {list(self.df.columns)}")
    
    def _build_encoders(self):
        """为不同类型特征创建编码器/归一化器"""
        # 1. 数值特征标准化器 (A0-A8: Z-score normalization)
        self.numeric_scalers = {}
        for feature in self.numeric_features:
            if feature in self.df.columns:
                scaler = StandardScaler()
                numeric_data = pd.to_numeric(self.df[feature], errors='coerce').fillna(0)
                scaler.fit(numeric_data.values.reshape(-1, 1))
                self.numeric_scalers[feature] = scaler
        
        # 2. 分类特征编码器 (One-hot encoding)
        self.categorical_encoders = {}
        self.categorical_dims = {}
        
        for feature in self.categorical_features:
            if feature in self.df.columns:
                # 处理特殊情况
                if feature == 'All_Metals(金属种类)':
                    # 金属种类需要特殊处理，支持多种金属
                    unique_metals = set()
                    for metals in self.df[feature]:
                        if isinstance(metals, str) and metals:
                            # 分割多个金属（如"Mo,Co"）
                            for metal in metals.split(','):
                                unique_metals.add(metal.strip())
                    self.metal_vocab = {metal: i for i, metal in enumerate(sorted(unique_metals))}
                    self.categorical_dims[feature] = len(unique_metals)
                
                elif feature == 'Open_Metal_Sites(缺陷位点)':
                    # 缺陷位点也可能有多种金属
                    unique_sites = set()
                    for sites in self.df[feature]:
                        if isinstance(sites, str) and sites:
                            for site in sites.split(','):
                                unique_sites.add(site.strip())
                    self.oms_vocab = {site: i for i, site in enumerate(sorted(unique_sites))}
                    self.categorical_dims[feature] = len(unique_sites)
                
                else:
                    # 普通分类特征
                    encoder = LabelEncoder()
                    cat_data = self.df[feature].astype(str).fillna('unknown')
                    encoder.fit(cat_data)
                    self.categorical_encoders[feature] = encoder
                    self.categorical_dims[feature] = len(encoder.classes_)
        
        # 3. 配体特征编码器
        for ligand_name, attrs in self.ligand_features.items():
            for attr_name, attr_type in attrs.items():
                col_name = f"{ligand_name}.{attr_name}" if f"{ligand_name}.{attr_name}" in self.df.columns else attr_name
                
                if col_name in self.df.columns:
                    if attr_type == 'numeric':
                        # 配体长度：归一化到[0,1]
                        scaler = MinMaxScaler()
                        numeric_data = pd.to_numeric(self.df[col_name], errors='coerce').fillna(0)
                        scaler.fit(numeric_data.values.reshape(-1, 1))
                        self.numeric_scalers[f"{ligand_name}_{attr_name}"] = scaler
                    
                    elif attr_type == 'categorical':
                        # 配体结构和基团：One-hot编码
                        encoder = LabelEncoder()
                        cat_data = self.df[col_name].astype(str).fillna('unknown')
                        encoder.fit(cat_data)
                        self.categorical_encoders[f"{ligand_name}_{attr_name}"] = encoder
                        self.categorical_dims[f"{ligand_name}_{attr_name}"] = len(encoder.classes_)
        
        # 4. SMILES特征处理
        self._build_smiles_encoders()
    
    def _build_smiles_encoders(self):
        """构建SMILES编码器"""
        # 简单的字符级SMILES编码
        all_chars = set()
        for feature in self.smiles_features:
            if feature in self.df.columns:
                for smiles in self.df[feature]:
                    if isinstance(smiles, str) and smiles:
                        all_chars.update(list(smiles))
        
        # 添加特殊字符
        all_chars.update(['<PAD>', '<UNK>', '<START>', '<END>'])
        self.smiles_vocab = {char: i for i, char in enumerate(sorted(all_chars))}
        self.vocab_size = len(self.smiles_vocab)
        
        print(f"SMILES vocabulary size: {self.vocab_size}")
    
    def _build_neural_networks(self):
        """定义用于不同类型特征的 MLP / LSTM 编码结构"""
        # 1. 数值特征处理网络 (A0-A8)
        numeric_dim = len(self.numeric_features)
        for ligand_name, attrs in self.ligand_features.items():
            for attr_name, attr_type in attrs.items():
                if attr_type == 'numeric':
                    numeric_dim += 1
        
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        # 2. 分类特征嵌入层
        self.categorical_embeddings = nn.ModuleDict()
        total_cat_dim = 0
        
        for feature, dim in self.categorical_dims.items():
            embed_dim = min(50, (dim + 1) // 2)  # 动态嵌入维度
            self.categorical_embeddings[feature.replace('(', '').replace(')', '').replace(' ', '_')] = nn.Embedding(dim, embed_dim)
            total_cat_dim += embed_dim
        
        self.categorical_encoder = nn.Sequential(
            nn.Linear(total_cat_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        # 3. SMILES编码器
        self.smiles_embedding = nn.Embedding(self.vocab_size, self.smiles_embedding_dim)
        self.smiles_encoder = nn.LSTM(
            self.smiles_embedding_dim,
            self.smiles_embedding_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # 每个SMILES特征的处理
        self.smiles_processors = nn.ModuleDict()
        for feature in self.smiles_features:
            self.smiles_processors[feature.replace('(', '').replace(')', '').replace(' ', '_')] = nn.Sequential(
                nn.Linear(self.smiles_embedding_dim * 2, self.smiles_embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # 4. 配体特征融合器
        ligand_feature_dim = 0
        for ligand_name in ['Ligand_1', 'Ligand_2', 'Ligand_3']:
            # 每个配体：SMILES + length + structures + groups
            ligand_feature_dim += self.smiles_embedding_dim  # SMILES
            ligand_feature_dim += 1  # length
            # structures 和 groups 的嵌入维度
            for attr in ['structures', 'groups']:
                key = f"{ligand_name}_{attr}"
                if key in self.categorical_dims:
                    embed_dim = min(50, (self.categorical_dims[key] + 1) // 2)
                    ligand_feature_dim += embed_dim
        
        self.ligand_encoder = nn.Sequential(
            nn.Linear(ligand_feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        # 5. 最终融合网络
        final_input_dim = (
            self.hidden_dim // 2 +  # numeric features
            self.hidden_dim // 2 +  # categorical features  
            len(self.smiles_features) * self.smiles_embedding_dim +  # SMILES features
            self.hidden_dim // 2    # ligand features
        )
        
        self.final_encoder = nn.Sequential(
            nn.Linear(final_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.global_dim)
        )
        # ---------------------------------------------------------
        # SMILES 编码函数
        # ---------------------------------------------------------
    def _encode_smiles(self, smiles: str, max_length: int = 100) -> torch.Tensor:
        """把 SMILES 字符串转为索引序列"""
        if not isinstance(smiles, str) or not smiles:
            return torch.zeros(max_length, dtype=torch.long)
        
        # 字符级编码
        encoded = []
        for char in smiles[:max_length-1]:
            encoded.append(self.smiles_vocab.get(char, self.smiles_vocab['<UNK>']))
        
        # 填充
        while len(encoded) < max_length:
            encoded.append(self.smiles_vocab['<PAD>'])
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def _encode_metals(self, metals_str: str) -> torch.Tensor:
        """编码金属种类（支持多种金属）"""
        if not isinstance(metals_str, str) or not metals_str:
            return torch.zeros(len(self.metal_vocab))
        
        encoding = torch.zeros(len(self.metal_vocab))
        for metal in metals_str.split(','):
            metal = metal.strip()
            if metal in self.metal_vocab:
                encoding[self.metal_vocab[metal]] = 1.0
        
        return encoding
    
    def _encode_oms(self, oms_str: str) -> torch.Tensor:
        """编码开放金属位点"""
        if not isinstance(oms_str, str) or not oms_str:
            return torch.zeros(len(self.oms_vocab))
        
        encoding = torch.zeros(len(self.oms_vocab))
        for site in oms_str.split(','):
            site = site.strip()
            if site in self.oms_vocab:
                encoding[self.oms_vocab[site]] = 1.0
        
        return encoding
    # ---------------------------------------------------------
    # 主编码函数
    # ---------------------------------------------------------
    def encode_mof_by_name(self, mof_name: str) -> torch.Tensor:
        """
        根据 MOF 名称查找 Excel 表中对应行，提取并编码所有特征，
        最后融合为一个 global embedding 向量。
        """
        if mof_name not in self.name_to_idx:
            print(f"Warning: MOF {mof_name} not found in database")
            return torch.zeros(self.global_dim, device=self.device)
        
        idx = self.name_to_idx[mof_name]
        row = self.df.iloc[idx]
        
        feature_parts = []
        
        # 1. 编码数值特征 (A0-A8)
        numeric_features = []
        for feature in self.numeric_features:
            if feature in self.df.columns:
                value = pd.to_numeric(row[feature], errors='coerce')
                if pd.isna(value):
                    value = 0.0
                if feature in self.numeric_scalers:
                    value = self.numeric_scalers[feature].transform([[value]])[0][0]
                numeric_features.append(value)
        
        # 添加配体数值特征
        for ligand_name, attrs in self.ligand_features.items():
            for attr_name, attr_type in attrs.items():
                if attr_type == 'numeric':
                    col_name = f"{ligand_name}.{attr_name}" if f"{ligand_name}.{attr_name}" in self.df.columns else attr_name
                    if col_name in self.df.columns:
                        value = pd.to_numeric(row[col_name], errors='coerce')
                        if pd.isna(value):
                            value = 0.0
                        scaler_key = f"{ligand_name}_{attr_name}"
                        if scaler_key in self.numeric_scalers:
                            value = self.numeric_scalers[scaler_key].transform([[value]])[0][0]
                        numeric_features.append(value)
        
        if numeric_features:
            numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32, device=self.device)
            numeric_encoded = self.numeric_encoder(numeric_tensor.unsqueeze(0)).squeeze(0)
            feature_parts.append(numeric_encoded)
        
        # 2. 编码分类特征
        categorical_features = []
        
        for feature in self.categorical_features:
            if feature in self.df.columns:
                feature_key = feature.replace('(', '').replace(')', '').replace(' ', '_')
                
                if feature == 'All_Metals(金属种类)':
                    # 多热编码
                    encoding = self._encode_metals(str(row[feature]))
                    categorical_features.append(encoding)
                
                elif feature == 'Open_Metal_Sites(缺陷位点)':
                    # 多热编码
                    encoding = self._encode_oms(str(row[feature]))
                    categorical_features.append(encoding)
                
                else:
                    # 普通分类特征
                    if feature in self.categorical_encoders:
                        try:
                            encoded_val = self.categorical_encoders[feature].transform([str(row[feature])])[0]
                            embedding = self.categorical_embeddings[feature_key](torch.tensor(encoded_val, device=self.device))
                            categorical_features.append(embedding)
                        except:
                            # 处理未见过的类别
                            embed_dim = self.categorical_embeddings[feature_key].embedding_dim
                            categorical_features.append(torch.zeros(embed_dim, device=self.device))
        
        # 添加配体分类特征
        for ligand_name, attrs in self.ligand_features.items():
            for attr_name, attr_type in attrs.items():
                if attr_type == 'categorical':
                    col_name = f"{ligand_name}.{attr_name}" if f"{ligand_name}.{attr_name}" in self.df.columns else attr_name
                    if col_name in self.df.columns:
                        encoder_key = f"{ligand_name}_{attr_name}"
                        embed_key = encoder_key
                        
                        if encoder_key in self.categorical_encoders:
                            try:
                                encoded_val = self.categorical_encoders[encoder_key].transform([str(row[col_name])])[0]
                                embedding = self.categorical_embeddings[embed_key](torch.tensor(encoded_val, device=self.device))
                                categorical_features.append(embedding)
                            except:
                                embed_dim = self.categorical_embeddings[embed_key].embedding_dim
                                categorical_features.append(torch.zeros(embed_dim, device=self.device))
        
        if categorical_features:
            categorical_tensor = torch.cat(categorical_features, dim=0)
            categorical_encoded = self.categorical_encoder(categorical_tensor.unsqueeze(0)).squeeze(0)
            feature_parts.append(categorical_encoded)
        
        # 3. 编码SMILES特征
        smiles_features = []
        for feature in self.smiles_features:
            if feature in self.df.columns:
                smiles_str = str(row[feature])
                if smiles_str and smiles_str != 'nan':
                    # 编码SMILES
                    encoded_smiles = self._encode_smiles(smiles_str).to(self.device)
                    embedded_smiles = self.smiles_embedding(encoded_smiles)
                    
                    # LSTM编码
                    lstm_out, (hidden, _) = self.smiles_encoder(embedded_smiles.unsqueeze(0))
                    # 使用最后一个隐藏状态
                    final_hidden = torch.cat([hidden[0], hidden[1]], dim=-1).squeeze(0)
                    
                    # 通过处理器
                    feature_key = feature.replace('(', '').replace(')', '').replace(' ', '_')
                    processed = self.smiles_processors[feature_key](final_hidden)
                    smiles_features.append(processed)
                else:
                    smiles_features.append(torch.zeros(self.smiles_embedding_dim, device=self.device))
        
        if smiles_features:
            smiles_tensor = torch.cat(smiles_features, dim=0)
            feature_parts.append(smiles_tensor)
        
        # 4. 配体特征融合
        ligand_features = []
        for ligand_name in ['Ligand_1', 'Ligand_2', 'Ligand_3']:
            ligand_parts = []
            
            # SMILES部分
            if ligand_name in self.df.columns:
                smiles_str = str(row[ligand_name])
                if smiles_str and smiles_str != 'nan':
                    encoded_smiles = self._encode_smiles(smiles_str).to(self.device)
                    embedded_smiles = self.smiles_embedding(encoded_smiles)
                    lstm_out, (hidden, _) = self.smiles_encoder(embedded_smiles.unsqueeze(0))
                    final_hidden = torch.cat([hidden[0], hidden[1]], dim=-1).squeeze(0)
                    
                    feature_key = ligand_name.replace('(', '').replace(')', '').replace(' ', '_')
                    processed = self.smiles_processors[feature_key](final_hidden)
                    ligand_parts.append(processed)
                else:
                    ligand_parts.append(torch.zeros(self.smiles_embedding_dim, device=self.device))
            
            # 数值特征（length）
            length_col = f"{ligand_name}.length" if f"{ligand_name}.length" in self.df.columns else "length"
            if length_col in self.df.columns:
                length_val = pd.to_numeric(row[length_col], errors='coerce')
                if pd.isna(length_val):
                    length_val = 0.0
                scaler_key = f"{ligand_name}_length"
                if scaler_key in self.numeric_scalers:
                    length_val = self.numeric_scalers[scaler_key].transform([[length_val]])[0][0]
                ligand_parts.append(torch.tensor([length_val], device=self.device))
            
            # 分类特征（structures, groups）
            for attr in ['structures', 'groups']:
                col_name = f"{ligand_name}.{attr}" if f"{ligand_name}.{attr}" in self.df.columns else attr
                if col_name in self.df.columns:
                    encoder_key = f"{ligand_name}_{attr}"
                    if encoder_key in self.categorical_encoders:
                        try:
                            encoded_val = self.categorical_encoders[encoder_key].transform([str(row[col_name])])[0]
                            embedding = self.categorical_embeddings[encoder_key](torch.tensor(encoded_val, device=self.device))
                            ligand_parts.append(embedding)
                        except:
                            embed_dim = self.categorical_embeddings[encoder_key].embedding_dim
                            ligand_parts.append(torch.zeros(embed_dim, device=self.device))
            
            if ligand_parts:
                ligand_feature = torch.cat(ligand_parts, dim=0)
                ligand_features.append(ligand_feature)
        
        if ligand_features:
            ligand_tensor = torch.cat(ligand_features, dim=0)
            ligand_encoded = self.ligand_encoder(ligand_tensor.unsqueeze(0)).squeeze(0)
            feature_parts.append(ligand_encoded)
        
        # 5. 最终融合
        if feature_parts:
            final_features = torch.cat(feature_parts, dim=0)
            global_embedding = self.final_encoder(final_features.unsqueeze(0)).squeeze(0)
            return global_embedding
        else:
            return torch.zeros(self.global_dim, device=self.device)
    
    def forward(self, mof_names: Union[str, List[str]]) -> torch.Tensor:
        """
        前向传播
        Args:
            mof_names: MOF名称或名称列表
        Returns:
            全局嵌入张量 [batch_size, global_dim]
        """
        if isinstance(mof_names, str):
            mof_names = [mof_names]
        
        embeddings = []
        for name in mof_names:
            embedding = self.encode_mof_by_name(name)
            embeddings.append(embedding)
        
        return torch.stack(embeddings, dim=0)
    
    def save_encoders(self, save_path: str):
        """保存编码器"""
        encoders_dict = {
            'numeric_scalers': self.numeric_scalers,
            'categorical_encoders': self.categorical_encoders,
            'categorical_dims': self.categorical_dims,
            'smiles_vocab': self.smiles_vocab,
            'metal_vocab': getattr(self, 'metal_vocab', {}),
            'oms_vocab': getattr(self, 'oms_vocab', {}),
            'name_to_idx': self.name_to_idx,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(encoders_dict, f)
        
        print(f"Encoders saved to {save_path}")
    
    def load_encoders(self, load_path: str):
        """加载编码器"""
        with open(load_path, 'rb') as f:
            encoders_dict = pickle.load(f)
        
        self.numeric_scalers = encoders_dict['numeric_scalers']
        self.categorical_encoders = encoders_dict['categorical_encoders']
        self.categorical_dims = encoders_dict['categorical_dims']
        self.smiles_vocab = encoders_dict['smiles_vocab']
        self.metal_vocab = encoders_dict.get('metal_vocab', {})
        self.oms_vocab = encoders_dict.get('oms_vocab', {})
        self.name_to_idx = encoders_dict['name_to_idx']
        
        print(f"Encoders loaded from {load_path}")

# 使用示例
def test_mof_encoder():
    """测试MOF编码器"""
    # 初始化编码器
    encoder = MOFGlobalPropertyEncoder(
        excel_path='MOF_embedding_train-check580.xlsx',
        global_dim=128,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 测试单个MOF
    test_names = ['ABEFUL', 'ABESUX', 'ABEXEM']
    
    with torch.no_grad():
        embeddings = encoder(test_names)
        print(f"Generated embeddings shape: {embeddings.shape}")
        print(f"Sample embedding norm: {embeddings[0].norm().item():.4f}")
    
    # 保存编码器
    encoder.save_encoders('mof_encoders.pkl')
    
    return encoder

if __name__ == "__main__":
    encoder = test_mof_encoder()
