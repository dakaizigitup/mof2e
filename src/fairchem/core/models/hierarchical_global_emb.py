import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class GeometryEncoder(nn.Module):
    """
    Encoder for Group A: Textural & Geometric Properties (The "Space")
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) # [batch, output_dim]

class NodeTopologyEncoder(nn.Module):
    """
    Encoder for Group B: Metal Node & Topology (The "Backbone")
    Uses embeddings for categorical features and aggregates them.
    """
    def __init__(self, embedding_dims: Dict[str, Tuple[int, int]], output_dim: int = 64):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for key, (num_embeddings, embedding_dim) in embedding_dims.items():
            self.embeddings[key] = nn.Embedding(num_embeddings, embedding_dim)
            total_emb_dim += embedding_dim
            
        self.mlp = nn.Sequential(
            nn.Linear(total_emb_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        emb_list = []
        # Ensure deterministic order based on keys
        for key in sorted(self.embeddings.keys()):
            if key in inputs:
                x = inputs[key]
                # x shape: [batch] or [batch, max_tokens]
                emb = self.embeddings[key](x) # [batch, emb_dim] or [batch, max_tokens, emb_dim]
                
                if emb.dim() == 3:
                     # Multi-label case: Mean pooling over tokens (ignoring padding 0 which is mapped to something, wait 0 is usually padding)
                     # Assuming 0 is padding index for convenience, or we just trust mean
                     # For simplicity in this upgrade: strict mean pooling
                     mask = (x != 0).float().unsqueeze(-1) # [batch, max_tokens, 1]
                     sum_emb = (emb * mask).sum(dim=1)
                     count = mask.sum(dim=1).clamp(min=1)
                     emb = sum_emb / count
                     
                emb_list.append(emb)
            else:
                pass
        
        if not emb_list:
            return torch.zeros(inputs[list(inputs.keys())[0]].shape[0], self.mlp[-1].out_features, device=list(self.parameters())[0].device)
            
        concat = torch.cat(emb_list, dim=-1)
        return self.mlp(concat)

class LigandSetEncoder(nn.Module):
    """
    Encoder for Group C: Organic Linkers (The "Walls")
    Uses DeepSets approach: Shared Encoder -> Sum Pooling
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        # Shared encoder for single ligand
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, ligands: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            ligands: List of [batch, input_dim] tensors, one for each ligand.
        Returns:
            aggregated: [batch, output_dim]
        """
        encoded_list = [self.shared_encoder(lig) for lig in ligands]
        # Stack and sum (DeepSets aggregation)
        stacked = torch.stack(encoded_list, dim=1) # [batch, num_ligands, output_dim]
        pooled = stacked.sum(dim=1) # [batch, output_dim]
        return pooled

class MOFHierarchicalEncoder(nn.Module):
    """
    Hierarchical Grouped Expert Encoder for MOF Global Properties.
    """
    def __init__(
        self,
        excel_path: str,
        device: str = 'cuda'
    ):
        super().__init__()
        self.excel_path = excel_path
        self.device = device
        
        # Load and preprocess data
        self.df = pd.read_excel(excel_path).fillna('')
        self._preprocess_data()
        self._build_encoders()
        
        # Feature Groups Definitions
        self.geo_features = [
            'LCD(最大腔体直径)', 'PLD(最小孔径)', 'LFPD(最大自由球体直径)',
            'ASA_m2_cm3(比表面积)', 'NASA_m2_cm3(不可接触表面积)',
            'AV_VF(孔隙率)', 'AV_cm3_g(可接触体积)', 'NAV_cm3_g(不可接触体积)'
        ]
        
        self.node_features = [
            'All_Metals(金属种类)', 'Has_OMS(是否有缺陷)', 
            'Open_Metal_Sites(缺陷位点)', 'Topologies(拓扑)'
        ]
        
        # Ligand features are handled specially (Ligand_1/2/3)
        self.ligand_subfeatures = ['length', 'structures', 'groups']
        
        # 1. Geometry Encoder
        self.geo_encoder = GeometryEncoder(input_dim=len(self.geo_features))
        
        # 2. Node/Topology Encoder
        # We need to pre-calculate embedding dimensions
        embedding_dims = {}
        for f in self.node_categorical_dims:
            dim = self.node_categorical_dims[f]
            embedding_dims[self._sanitize_key(f)] = (dim, min(50, (dim + 1) // 2))
        self.node_encoder = NodeTopologyEncoder(embedding_dims)
        
        # 3. Ligand Set Encoder
        # Input dim = length(1) + structures_emb + groups_emb + smiles_emb
        self.ligand_struct_emb_dim = 16
        self.ligand_group_emb_dim = 16
        self.ligand_smiles_dim = 64
        
        # Embeddings for ligand categorical features
        self.ligand_embeddings = nn.ModuleDict() 
        # Note: We share embeddings across Ligand 1/2/3 positions for invariant features?
        # Ideally yes, "structures" vocab should be shared.
        # Let's build a unified vocab for ligand structures and groups.
        self._build_ligand_vocabs()
        
        self.ligand_input_dim = 1 + self.ligand_struct_emb_dim + self.ligand_group_emb_dim + self.ligand_smiles_dim
        self.ligand_encoder = LigandSetEncoder(input_dim=self.ligand_input_dim)
        
        # 4. SMILES Encoder (Shared) - Re-added
        self.smiles_embedding = nn.Embedding(self.vocab_size, 32)
        self.smiles_rnn = nn.LSTM(32, 32, batch_first=True, bidirectional=True) # Output 64
        
        # 5. Gating Network (Judge)
        self.fusion_input_dim = 64 + 64 + 64 # geo + node + ligand
        self.gating_net = nn.Sequential(
            nn.Linear(self.fusion_input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3), # 3 experts: Geo, Node, Ligand
            nn.Softmax(dim=-1)
        )
        
        self.to(device)
        self.eval() # defaulted to eval

    def _preprocess_data(self):
        # Name mapping
        self.name_to_idx = {}
        for idx, row in self.df.iterrows():
            name = str(row['Name']).strip().upper().split('_')[0]
            self.name_to_idx[name] = idx
            
    def _sanitize_key(self, name: str) -> str:
        return name.replace('(', '').replace(')', '').replace(' ', '_').replace('.', '_')

    def _build_encoders(self):
        # Numeric Scalers
        self.scalers = {}
        geo_features = [
            'LCD(最大腔体直径)', 'PLD(最小孔径)', 'LFPD(最大自由球体直径)',
            'ASA_m2_cm3(比表面积)', 'NASA_m2_cm3(不可接触表面积)',
            'AV_VF(孔隙率)', 'AV_cm3_g(可接触体积)', 'NAV_cm3_g(不可接触体积)',
            'length', 'length.1', 'length.2' # Ligand lengths
        ]
        log1p_features = {'LCD(最大腔体直径)', 'PLD(最小孔径)', 'LFPD(最大自由球体直径)',
            'ASA_m2_cm3(比表面积)', 'NASA_m2_cm3(不可接触表面积)',
            'AV_cm3_g(可接触体积)', 'NAV_cm3_g(不可接触体积)'}
            
        for f in geo_features:
            if f in self.df.columns:
                scaler = StandardScaler()
                data = pd.to_numeric(self.df[f], errors='coerce').fillna(0)
                if f in log1p_features:
                    data = np.log1p(np.maximum(data.values, 0))
                
                if hasattr(data, 'values'):
                    data = data.values
                scaler.fit(data.reshape(-1, 1))
                self.scalers[f] = scaler
                self.scalers[f+"_log1p"] = (f in log1p_features)

        # Node Categorical Encoders (Custom Vocab for Multi-label)
        self.node_categorical_encoders = {}
        self.node_categorical_dims = {}
        # Multi-label features
        self.multi_label_feats = {'All_Metals(金属种类)', 'Open_Metal_Sites(缺陷位点)'}
        node_cats = ['All_Metals(金属种类)', 'Has_OMS(是否有缺陷)', 'Open_Metal_Sites(缺陷位点)', 'Topologies(拓扑)']
        
        for f in node_cats:
            if f in self.df.columns:
                if f in self.multi_label_feats:
                    # Tokenize and build set
                    tokens = set()
                    for entry in self.df[f].astype(str):
                        for t in entry.split(','):
                            t = t.strip()
                            if t: tokens.add(t)
                    # Add PAD/UNK
                    tokens.add('<PAD>') # 0
                    tokens = sorted(list(tokens))
                    # Manual mapping
                    vocab = {t: i for i, t in enumerate(tokens)}
                    # Ensure <PAD> is 0
                    if '<PAD>' in vocab and vocab['<PAD>'] != 0:
                        # Swap key 0 and PAD
                        zero_token = tokens[0]
                        vocab['<PAD>'] = 0
                        vocab[zero_token] = vocab['<PAD>_idx_orig'] # minimal swap logic, or just rebuild
                        # Simple rebuild
                        tokens.remove('<PAD>')
                        tokens = ['<PAD>'] + tokens
                        vocab = {t: i for i, t in enumerate(tokens)}
                    
                    self.node_categorical_encoders[f] = vocab
                    self.node_categorical_dims[f] = len(vocab)
                else:
                    # Single label
                    enc = LabelEncoder()
                    self.df[f] = self.df[f].astype(str)
                    enc.fit(self.df[f])
                    self.node_categorical_encoders[f] = enc
                    self.node_categorical_dims[f] = len(enc.classes_)
            else:
                self.node_categorical_dims[f] = 1 # Dummy
                
        # SMILES Vocab
        chars = set()
        for f in ['Ligand_1', 'Ligand_2', 'Ligand_3']:
            if f in self.df.columns:
                for s in self.df[f].astype(str):
                   chars.update(list(s))
        chars.update(['<PAD>', '<UNK>', '<START>', '<END>'])
        self.smiles_vocab = {ch: i for i, ch in enumerate(sorted(chars))}
        self.vocab_size = len(self.smiles_vocab)

    def _build_ligand_vocabs(self):
        # Unified vocab for structures and groups across L1, L2, L3
        self.struct_enc = LabelEncoder()
        self.group_enc = LabelEncoder()
        
        all_structs = []
        all_groups = []
        
        for suffix in ['', '.1', '.2']:
            s_col = f'structures{suffix}'
            g_col = f'groups{suffix}'
            if s_col in self.df.columns:
                all_structs.extend(self.df[s_col].astype(str).tolist())
            if g_col in self.df.columns:
                all_groups.extend(self.df[g_col].astype(str).tolist())
                
        self.struct_enc.fit(all_structs + ['<UNK>'])
        self.group_enc.fit(all_groups + ['<UNK>'])
        
        self.total_structs = len(self.struct_enc.classes_)
        self.total_groups = len(self.group_enc.classes_)
        
        # Create shared embeddings
        self.ligand_embeddings['structures'] = nn.Embedding(self.total_structs, self.ligand_struct_emb_dim)
        self.ligand_embeddings['groups'] = nn.Embedding(self.total_groups, self.ligand_group_emb_dim)

    def _encode_smiles(self, smiles: str) -> torch.Tensor:
        if not smiles or smiles == 'nan':
            return torch.zeros(64, device=self.device) # 64 is output of bidirectional LSTM(32)
        
        ids = [self.smiles_vocab.get(c, self.smiles_vocab['<UNK>']) for c in smiles[:128]]
        if not ids: return torch.zeros(64, device=self.device)
        
        tensor = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        emb = self.smiles_embedding(tensor)
        _, (h, _) = self.smiles_rnn(emb)
        # h: [2, 1, 32] -> cat -> [1, 64]
        return torch.cat([h[0], h[1]], dim=-1).squeeze(0)

    def forward(self, mof_names: List[str]) -> torch.Tensor:
        # Batch accumulation
        batch_geo = []
        batch_node_cats = {self._sanitize_key(k): [] for k in self.node_categorical_encoders}
        
        batch_l1, batch_l2, batch_l3 = [], [], []
        
        for name in mof_names:
            norm_name = str(name).strip().upper().split('_')[0]
            if norm_name not in self.name_to_idx:
                # Fallback zero
                batch_geo.append(torch.zeros(len(self.geo_features), device=self.device))
                for k in batch_node_cats: batch_node_cats[k].append(torch.tensor(0, device=self.device))
                zeros_lig = torch.zeros(self.ligand_input_dim, device=self.device)
                batch_l1.append(zeros_lig)
                batch_l2.append(zeros_lig)
                batch_l3.append(zeros_lig)
                continue
                
            idx = self.name_to_idx[norm_name]
            row = self.df.iloc[idx]
            
            # 1. Geometry
            g_vals = []
            for f in self.geo_features:
                val = pd.to_numeric(row.get(f, 0), errors='coerce')
                if pd.isna(val): val = 0.0
                if self.scalers.get(f+"_log1p", False):
                    val = float(np.log1p(max(val, 0)))
                if f in self.scalers:
                    try: val = self.scalers[f].transform([[val]])[0][0]
                    except: pass
                g_vals.append(val)
            batch_geo.append(torch.tensor(g_vals, dtype=torch.float32, device=self.device))
            
            # 2. Node
            for f, enc in self.node_categorical_encoders.items():
                val = str(row.get(f, ''))
                
                if f in self.multi_label_feats:
                    # Multi-label: split -> lookup -> tensor
                    ids = []
                     # Use the dict vocab
                    vocab = enc
                    for t in val.split(','):
                        t = t.strip()
                        if t in vocab: ids.append(vocab[t])
                    if not ids: ids = [vocab.get('<PAD>', 0)]
                    
                    # Store as tensor. We will need to pad later or now. 
                    # Let's just output a tensor [L]. NodeTopologyEncoder will handle list of tensors?
                    # Wait, Encoder expects inputs[key] to be batched tensor.
                    # We need to pad in the batch construction loop below.
                    # For now append raw list of ids
                    batch_node_cats[self._sanitize_key(f)].append(torch.tensor(ids, dtype=torch.long, device=self.device))
                
                else:
                    # Single-label
                    try: label = enc.transform([val])[0]
                    except: label = 0
                    batch_node_cats[self._sanitize_key(f)].append(torch.tensor(label, dtype=torch.long, device=self.device))
                
            # 3. Ligands
            lig_vectors = []
            for i, suffix in enumerate(['', '.1', '.2']):
                lig_name = f"Ligand_{i+1}"
                # Length
                l_col = f"length{suffix}"
                l_val = pd.to_numeric(row.get(l_col, 0), errors='coerce')
                if pd.isna(l_val): l_val = 0.0
                if l_col in self.scalers:
                   try: l_val = self.scalers[l_col].transform([[l_val]])[0][0]
                   except: pass
                l_tens = torch.tensor([l_val], dtype=torch.float32, device=self.device)
                
                # Structure
                s_col = f"structures{suffix}"
                s_val = str(row.get(s_col, '<UNK>'))
                try: s_idx = self.struct_enc.transform([s_val])[0]
                except: s_idx = 0
                s_emb = self.ligand_embeddings['structures'](torch.tensor(s_idx, device=self.device))
                
                # Group
                g_col = f"groups{suffix}"
                g_val = str(row.get(g_col, '<UNK>'))
                try: g_idx = self.group_enc.transform([g_val])[0]
                except: g_idx = 0
                g_emb = self.ligand_embeddings['groups'](torch.tensor(g_idx, device=self.device))
                
                # SMILES
                smi = str(row.get(lig_name, ''))
                smi_emb = self._encode_smiles(smi)
                
                # Concat
                lig_vec = torch.cat([l_tens, s_emb, g_emb, smi_emb], dim=0)
                lig_vectors.append(lig_vec)
                
            batch_l1.append(lig_vectors[0])
            batch_l2.append(lig_vectors[1])
            batch_l3.append(lig_vectors[2])
            
        # Stack batches
        geo_tensor = torch.stack(batch_geo)
        
        node_inputs = {}
        for k, v_list in batch_node_cats.items():
            # Check if elements are scalars or vectors
            if v_list[0].dim() == 0:
                node_inputs[k] = torch.stack(v_list)
            else:
                # Pad to max len in batch
                max_len = max([t.size(0) for t in v_list])
                padded_list = []
                for t in v_list:
                    pad_size = max_len - t.size(0)
                    if pad_size > 0:
                        # Pad with 0 (assuming <PAD> is 0)
                        t = torch.cat([t, torch.zeros(pad_size, dtype=torch.long, device=self.device)], dim=0)
                    padded_list.append(t)
                node_inputs[k] = torch.stack(padded_list)

        l1_tensor = torch.stack(batch_l1)
        l2_tensor = torch.stack(batch_l2)
        l3_tensor = torch.stack(batch_l3)
        
        # Encode
        h_geo = self.geo_encoder(geo_tensor)
        h_node = self.node_encoder(node_inputs)
        h_ligand = self.ligand_encoder([l1_tensor, l2_tensor, l3_tensor])
        
        # Gated Fusion
        # 1. Stack experts
        experts = torch.stack([h_geo, h_node, h_ligand], dim=1) # [batch, 3, 64]
        
        # 2. Calculate weights from context
        concat_context = torch.cat([h_geo, h_node, h_ligand], dim=-1) # [batch, 192]
        weights = self.gating_net(concat_context) # [batch, 3]
        
        # 3. Weighted Sum
        # weights: [batch, 3] -> [batch, 3, 1] for broadcasting
        weights = weights.unsqueeze(-1)
        
        final_embedding = (experts * weights).sum(dim=1) # [batch, 64]
        
        return final_embedding
