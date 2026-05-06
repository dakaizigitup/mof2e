import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 尝试导入RDKit用于分子特征提取
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Using basic SMILES encoding.")
    RDKIT_AVAILABLE = False


class LearnableSMILESEncoder:
    """可学习的SMILES编码器"""
    
    def __init__(self, max_length: int = 100, use_molecular_features: bool = True):
        self.max_length = max_length
        self.use_molecular_features = use_molecular_features
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.is_fitted = False
        self.molecular_feature_dim = 200 if use_molecular_features and RDKIT_AVAILABLE else 0
    
    def _extract_molecular_features(self, smiles: str) -> np.ndarray:
        """提取分子描述符特征"""
        if not RDKIT_AVAILABLE or not smiles or smiles.strip() == '':
            return np.zeros(self.molecular_feature_dim)
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.molecular_feature_dim)
            
            features = []
            
            # 基本分子属性
            features.extend([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.RingCount(mol),
                Descriptors.FractionCsp3(mol),
            ])
            
            # Morgan指纹
            morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
            features.extend(list(morgan_fp))
            
            # MACCS指纹
            maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            features.extend(list(maccs_fp))
            
            # 填充或截断到固定长度
            features = features[:self.molecular_feature_dim]
            while len(features) < self.molecular_feature_dim:
                features.append(0.0)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error extracting molecular features: {e}")
            return np.zeros(self.molecular_feature_dim)
    
    def fit(self, smiles_list: List[str]):
        """训练编码器，构建词汇表"""
        all_chars = set()
        for smiles in smiles_list:
            if isinstance(smiles, str) and smiles.strip():
                all_chars.update(smiles)
        
        # 添加特殊字符
        all_chars.add('<PAD>')
        all_chars.add('<UNK>')
        all_chars.add('<START>')
        all_chars.add('<END>')
        
        # 创建字符映射
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        self.is_fitted = True
    
    def encode_single(self, smiles: str) -> Dict[str, np.ndarray]:
        """编码单个SMILES"""
        if not self.is_fitted:
            raise ValueError("Encoder not fitted")
        
        # Token序列编码
        if not isinstance(smiles, str) or not smiles.strip():
            tokens = [self.char_to_idx['<PAD>']] * self.max_length
        else:
            smiles_with_markers = '<START>' + smiles + '<END>'
            tokens = []
            for char in smiles_with_markers[:self.max_length-2]:
                tokens.append(self.char_to_idx.get(char, self.char_to_idx['<UNK>']))
            
            while len(tokens) < self.max_length:
                tokens.append(self.char_to_idx['<PAD>'])
        
        result = {
            'tokens': np.array(tokens, dtype=np.int64),
            'attention_mask': np.array([1 if t != self.char_to_idx['<PAD>'] else 0 for t in tokens], dtype=np.float32)
        }
        
        # 分子特征
        if self.use_molecular_features:
            molecular_features = self._extract_molecular_features(smiles)
            result['molecular_features'] = molecular_features
        
        return result
    
    def encode_batch(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """批量编码SMILES"""
        batch_tokens = []
        batch_attention_masks = []
        batch_molecular_features = []
        
        for smiles in smiles_list:
            encoded = self.encode_single(smiles)
            batch_tokens.append(encoded['tokens'])
            batch_attention_masks.append(encoded['attention_mask'])
            
            if self.use_molecular_features:
                batch_molecular_features.append(encoded['molecular_features'])
        
        result = {
            'tokens': np.array(batch_tokens),
            'attention_mask': np.array(batch_attention_masks)
        }
        
        if self.use_molecular_features and batch_molecular_features:
            result['molecular_features'] = np.array(batch_molecular_features)
        
        return result


class MOFDataProcessor:
    """MOF数据预处理器 - 支持通过名称查找属性"""
    
    def __init__(self, use_molecular_features: bool = True):
        self.use_molecular_features = use_molecular_features
        self.is_fitted = False
        
        # 存储原始数据表
        self.data_table = None
        
        # 数值特征缩放器
        self.numerical_scaler = StandardScaler()
        
        # 分类特征编码器
        self.categorical_encoders = {}
        
        # SMILES编码器
        self.smiles_encoders = {}
        
        # 特征列定义
        self.numerical_cols = ['LCD(最大腔体直径)', 'PLD(最小孔径)', 'LFPD(最大自由球体直径)',
                              'ASA_m2_cm3(比表面积)', 'NASA_m2_cm3(不可接触表面积)', 
                              'AV_VF(孔隙率)', 'AV_cm3_g(可接触体积)', 'NAV_cm3_g(不可接触体积)']
        
        self.categorical_cols = ['All_Metals(金属种类)', 'Has_OMS(是否有缺陷)', 'Open_Metal_Sites(缺陷位点)',
                               'Topologies(拓扑)', 'Metals(金属节点)', 'Terminal Ligands(封端配体)']
        
        self.ligand_info = {
            'ligand1': {'smiles': 'Ligand_1', 'length': 'length1', 'structures': 'structures1', 'groups': 'groups1'},
            'ligand2': {'smiles': 'Ligand_2', 'length': 'length2', 'structures': 'structures2', 'groups': 'groups2'},
            'ligand3': {'smiles': 'Ligand_3', 'length': 'length3', 'structures': 'structures3', 'groups': 'groups3'}
        }

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        df = df.copy()
        
        # 处理数值特征缺失值
        for col in self.numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        # 处理分类特征缺失值
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('unknown')
        
        # 处理配体特征缺失值
        for ligand_key, ligand_info in self.ligand_info.items():
            for feature_type, col in ligand_info.items():
                if col in df.columns:
                    if feature_type == 'smiles':
                        df[col] = df[col].astype(str).fillna('')
                    elif feature_type == 'length':
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    else:
                        df[col] = df[col].astype(str).fillna('unknown')
        
        return df

    def fit(self, df: pd.DataFrame):
        """训练预处理器"""
        # 保存数据表用于后续查找
        self.data_table = self._preprocess_data(df)
        self.data_table.set_index('Name', inplace=True)  # 设置Name为索引
        
        df = self._preprocess_data(df)
        
        # 1. 训练数值特征缩放器
        numerical_data = []
        for col in self.numerical_cols:
            if col in df.columns:
                numerical_data.append(df[col].values.reshape(-1, 1))
        
        if numerical_data:
            combined_numerical = np.concatenate(numerical_data, axis=1)
            self.numerical_scaler.fit(combined_numerical)
        
        # 2. 训练分类特征编码器
        for col in self.categorical_cols:
            if col in df.columns:
                encoder = LabelEncoder()
                if col in ['All_Metals(金属种类)', 'Metals(金属节点)', 'Open_Metal_Sites(缺陷位点)']:
                    # 多标签处理
                    all_values = set()
                    for val in df[col]:
                        if pd.notna(val) and val != 'unknown':
                            values = [v.strip() for v in str(val).split(',')]
                            all_values.update(values)
                    encoder.fit(list(all_values) + ['unknown'])
                else:
                    encoder.fit(df[col])
                self.categorical_encoders[col] = encoder
        
        # 3. 训练SMILES编码器
        for ligand_key, ligand_info in self.ligand_info.items():
            smiles_col = ligand_info['smiles']
            if smiles_col in df.columns:
                smiles_encoder = LearnableSMILESEncoder(
                    max_length=100,
                    use_molecular_features=self.use_molecular_features
                )
                smiles_list = df[smiles_col].fillna('').astype(str).tolist()
                smiles_encoder.fit(smiles_list)
                self.smiles_encoders[ligand_key] = smiles_encoder
                
                # structures和groups编码器
                for feature_type in ['structures', 'groups']:
                    col = ligand_info[feature_type]
                    if col in df.columns:
                        encoder = LabelEncoder()
                        encoder.fit(df[col].fillna('unknown'))
                        self.categorical_encoders[f'{ligand_key}_{feature_type}'] = encoder
        
        self.is_fitted = True

    def get_mof_data_by_name(self, names: Union[str, List[str]]) -> pd.DataFrame:
        """根据MOF名称获取对应的数据"""
        if not self.is_fitted:
            raise ValueError("必须先调用fit方法")
        
        if isinstance(names, str):
            names = [names]
        
        # 查找对应的数据行
        found_data = []
        missing_names = []
        
        for name in names:
            if name in self.data_table.index:
                row_data = self.data_table.loc[name].to_dict()
                row_data['Name'] = name  # 添加名称
                found_data.append(row_data)
            else:
                missing_names.append(name)
        
        if missing_names:
            print(f"Warning: 以下MOF名称未在数据表中找到: {missing_names}")
        
        if not found_data:
            raise ValueError("没有找到任何有效的MOF数据")
        
        # 转换为DataFrame
        result_df = pd.DataFrame(found_data)
        return result_df

    def transform_by_names(self, names: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """根据MOF名称转换数据"""
        # 获取对应的数据
        df = self.get_mof_data_by_name(names)
        
        # 使用原有的transform方法
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """转换数据"""
        if not self.is_fitted:
            raise ValueError("必须先调用fit方法")
        
        df = self._preprocess_data(df)
        result = {}
        
        # 1. 处理数值特征
        numerical_data = []
        for col in self.numerical_cols:
            if col in df.columns:
                numerical_data.append(df[col].values.reshape(-1, 1))
        
        if numerical_data:
            combined_numerical = np.concatenate(numerical_data, axis=1)
            scaled_numerical = self.numerical_scaler.transform(combined_numerical)
            result['numerical'] = scaled_numerical
        
        # 2. 处理分类特征
        for col in self.categorical_cols:
            if col in df.columns and col in self.categorical_encoders:
                encoder = self.categorical_encoders[col]
                
                if col in ['All_Metals(金属种类)', 'Metals(金属节点)', 'Open_Metal_Sites(缺陷位点)']:
                    # 多标签one-hot编码
                    one_hot_matrix = []
                    for val in df[col]:
                        one_hot_vec = np.zeros(len(encoder.classes_))
                        if pd.notna(val) and val != 'unknown':
                            values = [v.strip() for v in str(val).split(',')]
                            for v in values:
                                if v in encoder.classes_:
                                    idx = encoder.transform([v])[0]
                                    one_hot_vec[idx] = 1
                        one_hot_matrix.append(one_hot_vec)
                    result[col] = np.array(one_hot_matrix)
                else:
                    # 单标签one-hot编码
                    encoded = encoder.transform(df[col])
                    one_hot = np.eye(len(encoder.classes_))[encoded]
                    result[col] = one_hot
        
        # 3. 处理配体特征
        for ligand_key, ligand_info in self.ligand_info.items():
            if ligand_key in self.smiles_encoders:
                smiles_col = ligand_info['smiles']
                length_col = ligand_info['length']
                structures_col = ligand_info['structures']
                groups_col = ligand_info['groups']
                
                # SMILES编码
                if smiles_col in df.columns:
                    smiles_encoded = self.smiles_encoders[ligand_key].encode_batch(
                        df[smiles_col].fillna('').astype(str).tolist()
                    )
                    
                    # 长度特征
                    length_vals = df[length_col].fillna(0).values.reshape(-1, 1) if length_col in df.columns else np.zeros((len(df), 1))
                    
                    # structures编码
                    if f'{ligand_key}_structures' in self.categorical_encoders and structures_col in df.columns:
                        struct_encoded = self.categorical_encoders[f'{ligand_key}_structures'].transform(
                            df[structures_col].fillna('unknown')
                        )
                    else:
                        struct_encoded = np.zeros(len(df), dtype=int)
                    
                    # groups编码
                    if f'{ligand_key}_groups' in self.categorical_encoders and groups_col in df.columns:
                        groups_encoded = self.categorical_encoders[f'{ligand_key}_groups'].transform(
                            df[groups_col].fillna('unknown')
                        )
                    else:
                        groups_encoded = np.zeros(len(df), dtype=int)
                    
                    result[ligand_key] = {
                        'smiles_data': smiles_encoded,
                        'length': length_vals,
                        'structures': struct_encoded,
                        'groups': groups_encoded
                    }
        
        return result

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """训练并转换"""
        self.fit(df)
        return self.transform(df)

    def save(self, filepath: str):
        """保存预处理器"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'data_table': self.data_table,
                'numerical_scaler': self.numerical_scaler,
                'categorical_encoders': self.categorical_encoders,
                'smiles_encoders': self.smiles_encoders,
                'use_molecular_features': self.use_molecular_features,
                'is_fitted': self.is_fitted,
                'numerical_cols': self.numerical_cols,
                'categorical_cols': self.categorical_cols,
                'ligand_info': self.ligand_info
            }, f)

    def load(self, filepath: str):
        """加载预处理器"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.data_table = data['data_table']
            self.numerical_scaler = data['numerical_scaler']
            self.categorical_encoders = data['categorical_encoders']
            self.smiles_encoders = data['smiles_encoders']
            self.use_molecular_features = data.get('use_molecular_features', True)
            self.is_fitted = data['is_fitted']
            self.numerical_cols = data['numerical_cols']
            self.categorical_cols = data['categorical_cols']
            self.ligand_info = data['ligand_info']


class MOFEmbeddingNetwork(nn.Module):
    """
    MOF嵌入网络 - 输入MOF名称，输出固定维度的嵌入向量
    """
    
    def __init__(self, 
                 processor: MOFDataProcessor,
                 embedding_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.processor = processor
        self.embedding_dim = embedding_dim
        
        # 1. 数值特征投影
        if processor.numerical_cols:
            self.numerical_projection = nn.Sequential(
                nn.Linear(len(processor.numerical_cols), 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            numerical_dim = 64
        else:
            self.numerical_projection = None
            numerical_dim = 0
        
        # 2. 分类特征投影
        self.categorical_projections = nn.ModuleDict()
        categorical_total_dim = 0
        for col in processor.categorical_cols:
            if col in processor.categorical_encoders:
                vocab_size = len(processor.categorical_encoders[col].classes_)
                self.categorical_projections[col] = nn.Sequential(
                    nn.Linear(vocab_size, 32),
                    nn.LayerNorm(32),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                categorical_total_dim += 32
        
        # 3. SMILES嵌入模块
        self.smiles_embeddings = nn.ModuleDict()
        self.ligand_projections = nn.ModuleDict()
        ligand_total_dim = 0
        
        for ligand_key, smiles_encoder in processor.smiles_encoders.items():
            # SMILES Token嵌入
            smiles_embedding_dim = 128
            self.smiles_embeddings[ligand_key] = nn.Sequential(
                nn.Embedding(smiles_encoder.vocab_size, smiles_embedding_dim, padding_idx=0),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=smiles_embedding_dim,
                        nhead=8,
                        dim_feedforward=256,
                        dropout=dropout,
                        batch_first=True
                    ),
                    num_layers=2
                )
            )
            
            # 分子特征投影
            mol_feature_dim = 0
            if processor.use_molecular_features and smiles_encoder.molecular_feature_dim > 0:
                mol_feature_dim = 64
                setattr(self, f'{ligand_key}_mol_projection', nn.Sequential(
                    nn.Linear(smiles_encoder.molecular_feature_dim, mol_feature_dim),
                    nn.LayerNorm(mol_feature_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
            
            # structures和groups嵌入
            struct_dim = groups_dim = 0
            if f'{ligand_key}_structures' in processor.categorical_encoders:
                struct_vocab_size = len(processor.categorical_encoders[f'{ligand_key}_structures'].classes_)
                struct_dim = 16
                setattr(self, f'{ligand_key}_struct_embedding', nn.Embedding(struct_vocab_size, struct_dim))
            
            if f'{ligand_key}_groups' in processor.categorical_encoders:
                groups_vocab_size = len(processor.categorical_encoders[f'{ligand_key}_groups'].classes_)
                groups_dim = 16
                setattr(self, f'{ligand_key}_groups_embedding', nn.Embedding(groups_vocab_size, groups_dim))
            
            # 长度特征投影
            length_dim = 8
            setattr(self, f'{ligand_key}_length_projection', nn.Linear(1, length_dim))
            
            # 配体融合层
            ligand_input_dim = smiles_embedding_dim + mol_feature_dim + struct_dim + groups_dim + length_dim
            ligand_output_dim = 128
            self.ligand_projections[ligand_key] = nn.Sequential(
                nn.Linear(ligand_input_dim, ligand_output_dim),
                nn.LayerNorm(ligand_output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            ligand_total_dim += ligand_output_dim
        
        # 4. 全局融合层
        total_input_dim = numerical_dim + categorical_total_dim + ligand_total_dim
        self.global_fusion = nn.Sequential(
            nn.Linear(total_input_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, mof_names: Union[str, List[str], torch.Tensor]):
        """
        前向传播 - 支持直接输入MOF名称
        Args:
            mof_names: MOF名称，可以是字符串、字符串列表或预处理后的tensor数据
        Returns:
            embedding: [batch_size, embedding_dim] 的嵌入向量
        """
        # 如果输入是MOF名称，先转换为特征
        if isinstance(mof_names, (str, list)):
            batch_data = self.processor.transform_by_names(mof_names)
            batch_data = self._convert_to_tensors(batch_data)
        else:
            # 假设输入已经是预处理后的tensor数据
            batch_data = mof_names
        
        features = []
        
        # 1. 处理数值特征
        if 'numerical' in batch_data and self.numerical_projection is not None:
            numerical_emb = self.numerical_projection(batch_data['numerical'])
            features.append(numerical_emb)
        
        # 2. 处理分类特征
        for col in self.processor.categorical_cols:
            if col in batch_data and col in self.categorical_projections:
                cat_emb = self.categorical_projections[col](batch_data[col])
                features.append(cat_emb)
        
        # 3. 处理配体特征
        for ligand_key in self.processor.ligand_info.keys():
            if ligand_key in batch_data and ligand_key in self.smiles_embeddings:
                ligand_data = batch_data[ligand_key]
                ligand_features = []
                
                # SMILES嵌入
                tokens = ligand_data['smiles_data']['tokens']
                attention_mask = ligand_data['smiles_data']['attention_mask']
                
                # Token嵌入
                token_emb = self.smiles_embeddings[ligand_key][0](tokens)  # Embedding层
                
                # Transformer编码
                transformer_mask = (attention_mask == 0)  # True表示忽略
                transformer_output = self.smiles_embeddings[ligand_key][1](
                    token_emb, 
                    src_key_padding_mask=transformer_mask
                )
                
                # 池化
                attention_weights = attention_mask.unsqueeze(-1)
                pooled_smiles = (transformer_output * attention_weights).sum(dim=1) / (attention_weights.sum(dim=1) + 1e-8)
                ligand_features.append(pooled_smiles)
                
                # 分子特征
                if hasattr(self, f'{ligand_key}_mol_projection') and 'molecular_features' in ligand_data['smiles_data']:
                    mol_emb = getattr(self, f'{ligand_key}_mol_projection')(ligand_data['smiles_data']['molecular_features'])
                    ligand_features.append(mol_emb)
                
                # structures嵌入
                if hasattr(self, f'{ligand_key}_struct_embedding'):
                    struct_emb = getattr(self, f'{ligand_key}_struct_embedding')(ligand_data['structures'])
                    ligand_features.append(struct_emb)
                
                # groups嵌入
                if hasattr(self, f'{ligand_key}_groups_embedding'):
                    groups_emb = getattr(self, f'{ligand_key}_groups_embedding')(ligand_data['groups'])
                    ligand_features.append(groups_emb)
                
                # 长度特征
                length_emb = getattr(self, f'{ligand_key}_length_projection')(ligand_data['length'])
                ligand_features.append(length_emb)
                
                # 配体特征融合
                combined_ligand = torch.cat(ligand_features, dim=-1)
                ligand_emb = self.ligand_projections[ligand_key](combined_ligand)
                features.append(ligand_emb)
        
        # 4. 全局特征融合
        if features:
            combined_features = torch.cat(features, dim=-1)
            global_embedding = self.global_fusion(combined_features)
            return global_embedding
        else:
            raise ValueError("No valid features found")
    
    def _convert_to_tensors(self, data_dict: Dict) -> Dict:
        """将numpy数据转换为tensor"""
        device = next(self.parameters()).device
        tensor_dict = {}
        
        for key, value in data_dict.items():
            if isinstance(value, dict):
                tensor_dict[key] = {}
                for sub_key, sub_value in value.items():
                    if sub_key == 'smiles_data':
                        tensor_dict[key][sub_key] = {
                            'tokens': torch.tensor(sub_value['tokens'], dtype=torch.long).to(device),
                            'attention_mask': torch.tensor(sub_value['attention_mask'], dtype=torch.float32).to(device)
                        }
                        if 'molecular_features' in sub_value:
                            tensor_dict[key][sub_key]['molecular_features'] = torch.tensor(
                                sub_value['molecular_features'], dtype=torch.float32
                            ).to(device)
                    else:
                        tensor_dict[key][sub_key] = torch.tensor(sub_value).to(device)
            else:
                tensor_dict[key] = torch.tensor(value, dtype=torch.float32).to(device)
        
        return tensor_dict


class MOFEmbeddingDataset(torch.utils.data.Dataset):
    """MOF嵌入数据集"""
    
    def __init__(self, embedding_dict: Dict[str, np.ndarray]):
        self.data = embedding_dict
        
        # 获取数据长度
        first_key = list(embedding_dict.keys())[0]
        if isinstance(self.data[first_key], dict):
            self.length = len(list(self.data[first_key].values())[0])
        else:
            self.length = len(self.data[first_key])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        batch_data = {}
        
        for key, value in self.data.items():
            if isinstance(value, dict):
                # 配体特征
                batch_data[key] = {}
                for sub_key, sub_value in value.items():
                    if sub_key == 'smiles_data':
                        batch_data[key][sub_key] = {
                            'tokens': torch.tensor(sub_value['tokens'][idx], dtype=torch.long),
                            'attention_mask': torch.tensor(sub_value['attention_mask'][idx], dtype=torch.float32)
                        }
                        if 'molecular_features' in sub_value:
                            batch_data[key][sub_key]['molecular_features'] = torch.tensor(
                                sub_value['molecular_features'][idx], dtype=torch.float32
                            )
                    else:
                        batch_data[key][sub_key] = torch.tensor(sub_value[idx])
            else:
                # 数值或分类特征
                batch_data[key] = torch.tensor(value[idx], dtype=torch.float32)
        
        return batch_data


class MOFEmbeddingTrainer:
    """MOF嵌入网络训练器"""
    
    def __init__(self, 
                 network: MOFEmbeddingNetwork,
                 processor: MOFDataProcessor,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        
        self.network = network
        self.processor = processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            network.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        self.training_history = {
            'loss': [],
            'epoch': []
        }
    
    def contrastive_loss(self, embeddings, temperature=0.1):
        """对比学习损失函数"""
        batch_size = embeddings.size(0)
        
        # 计算相似度矩阵
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T) / temperature
        
        # 创建标签（对角线为正样本）
        labels = torch.arange(batch_size).to(self.device)
        
        # 计算对比损失
        loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.network.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in dataloader:
            # 将数据移到设备
            batch_data = self._move_to_device(batch_data)
            
            # 前向传播
            embeddings = self.network(batch_data)
            
            # 计算损失
            loss = self.contrastive_loss(embeddings)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _move_to_device(self, batch_data):
        """将批量数据移到设备"""
        device_data = {}
        for key, value in batch_data.items():
            if isinstance(value, dict):
                device_data[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        device_data[key][sub_key] = {}
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            device_data[key][sub_key][sub_sub_key] = sub_sub_value.to(self.device)
                    else:
                        device_data[key][sub_key] = sub_value.to(self.device)
            else:
                device_data[key] = value.to(self.device)
        return device_data
    
    def train(self, df: pd.DataFrame, 
              num_epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.2,
              save_path: str = None,
              save_every: int = 10):
        """
        训练MOF嵌入网络
        """
        
        print(f"开始训练MOF嵌入网络，设备: {self.device}")
        
        # 预处理数据
        embedding_dict = self.processor.transform(df)
        
        # 创建数据集
        dataset = MOFEmbeddingDataset(embedding_dict)
        
        # 划分训练集和验证集
        if validation_split > 0:
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
        else:
            train_dataset = dataset
            val_dataset = None
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        if val_dataset:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=0
            )
        
        # 训练循环
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            if val_dataset:
                val_loss = self.validate(val_loader)
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                current_loss = val_loss
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
                current_loss = train_loss
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.training_history['loss'].append(current_loss)
            self.training_history['epoch'].append(epoch + 1)
            
            # 保存最佳模型
            if current_loss < best_loss:
                best_loss = current_loss
                if save_path:
                    torch.save(self.network.state_dict(), f'{save_path}_best_network.pth')
            
            # 定期保存
            if save_path and (epoch + 1) % save_every == 0:
                torch.save(self.network.state_dict(), f'{save_path}_epoch_{epoch+1}_network.pth')
        
        print(f"训练完成！最佳损失: {best_loss:.4f}")
        
        return self.training_history
    
    def validate(self, dataloader):
        """验证"""
        self.network.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = self._move_to_device(batch_data)
                embeddings = self.network(batch_data)
                loss = self.contrastive_loss(embeddings)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def get_embeddings(self, mof_names: Union[str, List[str]]) -> np.ndarray:
        """根据MOF名称获取嵌入向量"""
        self.network.eval()
        
        with torch.no_grad():
            embeddings = self.network(mof_names)
            return embeddings.cpu().numpy()


def create_mof_embedding_system(df: pd.DataFrame, 
                               embedding_dim: int = 256,
                               use_molecular_features: bool = True,
                               save_path: str = None) -> Tuple[MOFDataProcessor, MOFEmbeddingNetwork]:
    """
    创建MOF嵌入系统
    """
    
    # 1. 创建并训练数据预处理器
    print("创建并训练数据预处理器...")
    processor = MOFDataProcessor(use_molecular_features=use_molecular_features)
    processor.fit(df)
    
    # 2. 创建嵌入网络
    print("创建MOF嵌入网络...")
    network = MOFEmbeddingNetwork(
        processor=processor,
        embedding_dim=embedding_dim,
        dropout=0.1
    )
    
    # 3. 保存系统
    if save_path:
        print(f"保存预处理器到 {save_path}_processor.pkl")
        processor.save(f'{save_path}_processor.pkl')
        
        print(f"保存网络到 {save_path}_network.pth")
        torch.save(network.state_dict(), f'{save_path}_network.pth')
    
    return processor, network


def load_mof_embedding_system(processor_path: str, 
                             network_path: str) -> Tuple[MOFDataProcessor, MOFEmbeddingNetwork]:
    """
    加载MOF嵌入系统
    """
    
    # 加载预处理器
    processor = MOFDataProcessor()
    processor.load(processor_path)
    
    # 创建并加载网络
    network = MOFEmbeddingNetwork(
        processor=processor,
        embedding_dim=256,
        dropout=0.1
    )
    network.load_state_dict(torch.load(network_path, map_location='cpu'))
    
    return processor, network


def get_mof_embedding(processor_path: str,
                     network_path: str,
                     mof_names: Union[str, List[str]]) -> np.ndarray:
    """
    根据MOF名称获取嵌入向量的便捷函数
    
    Args:
        processor_path: 预处理器路径
        network_path: 网络路径
        mof_names: MOF名称，可以是单个字符串或字符串列表
    
    Returns:
        embeddings: [N, embedding_dim] 的嵌入矩阵
    """
    
    # 加载系统
    processor, network = load_mof_embedding_system(processor_path, network_path)
    
    # 设置为评估模式
    network.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    
    # 获取嵌入
    with torch.no_grad():
        embeddings = network(mof_names)
        return embeddings.cpu().numpy()


# 使用示例和测试
if __name__ == "__main__":
    # 读取数据
    df = pd.read_excel('MOF_embedding_train-check580.xlsx')
    
    print("=== 创建MOF嵌入系统 ===")
    
    # 1. 创建系统
    processor, network = create_mof_embedding_system(
        df=df,
        embedding_dim=256,
        use_molecular_features=True,
        save_path='mof_embedding_system'
    )
    
    print(f"网络参数数量: {sum(p.numel() for p in network.parameters()):,}")
    
    # 2. 创建训练器
    trainer = MOFEmbeddingTrainer(
        network=network,
        processor=processor,
        learning_rate=1e-3,
        weight_decay=1e-4
    )
    
    # 3. 训练网络
    print("\n=== 开始训练 ===")
    history = trainer.train(
        df=df,
        num_epochs=50,
        batch_size=16,
        validation_split=0.2,
        save_path='mof_embedding_system',
        save_every=10
    )
    
    # 4. 测试通过名称获取嵌入
    print("\n=== 测试通过名称获取嵌入 ===")
    
    # 测试单个MOF
    single_mof_name = "ABEFUL"
    single_embedding = trainer.get_embeddings(single_mof_name)
    print(f"单个MOF '{single_mof_name}' 嵌入形状: {single_embedding.shape}")
    print(f"嵌入向量前10个值: {single_embedding[0][:10]}")
    
    # 测试多个MOF
    multiple_mof_names = ["ABEFUL", "ABESUX", "ABEXEM"]
    multiple_embeddings = trainer.get_embeddings(multiple_mof_names)
    print(f"多个MOF {multiple_mof_names} 嵌入形状: {multiple_embeddings.shape}")
    
    # 5. 测试便捷函数
    print("\n=== 测试便捷函数 ===")
    convenience_embeddings = get_mof_embedding(
        processor_path='mof_embedding_system_processor.pkl',
        network_path='mof_embedding_system_best_network.pth',
        mof_names=["ABEFUL", "ABESUX"]
    )
    print(f"便捷函数获取的嵌入形状: {convenience_embeddings.shape}")
    
    # 6. 演示网络的直接使用
    print("\n=== 演示网络直接使用 ===")
    network.eval()
    with torch.no_grad():
        # 直接传入名称
        direct_embeddings = network("ABEFUL")
        print(f"直接使用网络的嵌入形状: {direct_embeddings.shape}")
        
        # 传入名称列表
        direct_batch_embeddings = network(["ABEFUL", "ABESUX"])
        print(f"直接使用网络（批量）的嵌入形状: {direct_batch_embeddings.shape}")
    
    print("\n=== MOF嵌入系统创建完成 ===")
    print("\n使用方法:")
    print("1. 训练: 使用 MOFEmbeddingTrainer 类")
    print("2. 便捷推理: 使用 get_mof_embedding 函数")
    print("3. 直接使用: network('MOF_NAME') 或 network(['MOF1', 'MOF2'])")
    print("4. 集成到其他模型: 在forward中调用 network(mof_names)")
    
    # 展示如何在其他模型中使用
    print("\n=== 集成示例 ===")
    
    class YourDownstreamModel(nn.Module):
        def __init__(self, mof_embedding_network, output_dim=1):
            super().__init__()
            self.mof_embedding = mof_embedding_network
            self.predictor = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, output_dim)
            )
        
        def forward(self, mof_names):
            # 获取MOF嵌入
            mof_embeddings = self.mof_embedding(mof_names)  # [batch_size, 256]
            
            # 进行下游预测
            predictions = self.predictor(mof_embeddings)  # [batch_size, output_dim]
            
            return predictions
    
    # 创建下游模型
    downstream_model = YourDownstreamModel(network, output_dim=1)
    
    # 测试下游模型
    with torch.no_grad():
        test_predictions = downstream_model(["ABEFUL", "ABESUX"])
        print(f"下游模型预测结果形状: {test_predictions.shape}")
        print(f"预测值: {test_predictions.squeeze().tolist()}")
