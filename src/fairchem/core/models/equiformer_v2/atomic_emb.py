import re
import torch
import torch.nn as nn
from typing import List, Tuple


def _safe_get(x, default=0.0):
    try:
        if x is None:
            return default
        if callable(x):
            x = x()
        return float(x) if x is not None else default
    except Exception:
        return default


class AtomPropertyEncoder(nn.Module):
    """
    预计算 1..max_Z 的元素表性质，构建标准化后的数值表，并通过小MLP映射到 out_dim。
    依赖 mendeleev（在 __init__ 时一次性读取，forward O(1) 查表）。
    输出用于节点标量通道融合（建议 out_dim=128）。
    """
    def __init__(self, max_Z: int = 100, out_dim: int = 128, dim_group: int = 32, dim_period: int = 32, dim_tag: int = 32):
        super().__init__()
        self.max_Z = int(max_Z)
        self.out_dim = int(out_dim)

        # Categorical features for Embedding
        self.cat_feature_names: List[Tuple[str, str]] = [
            ("group", "group_id"),
            ("period", "period"),
        ]
        
        # Numerical features
        self.num_feature_names: List[Tuple[str, str]] = [
            ("atomic_radius", "atomic_radius"),
            ("atomic_volume", "atomic_volume"),
            ("atomic_weight", "atomic_weight"),
            ("electron_affinity", "electron_affinity"),
            ("electronegativity_allen", "electronegativity_allen"),
            ("nvalence", "nvalence"),
        ]

        # Init lookup tables
        group_table = []
        period_table = []
        num_table = []

        try:
            from mendeleev import element
            for Z in range(0, self.max_Z + 1):
                if Z == 0:
                    group_table.append(0)
                    period_table.append(0)
                    num_table.append([0.0] * len(self.num_feature_names))
                    continue
                
                el = element(Z)
                
                # Group
                g = _safe_get(getattr(el, "group_id", 0), 0)
                group_table.append(int(g) if g > 0 else 0)
                
                # Period
                p = _safe_get(getattr(el, "period", 0), 0)
                period_table.append(int(p) if p > 0 else 0)
                
                # Num feats
                vals = []
                for name, attr in self.num_feature_names:
                    v = _safe_get(getattr(el, attr, None), 0.0)
                    vals.append(v)
                num_table.append(vals)

        except Exception as e:
            # Fallback
            group_table = [0] * (self.max_Z + 1)
            period_table = [0] * (self.max_Z + 1)
            num_table = [[0.0] * len(self.num_feature_names) for _ in range(self.max_Z + 1)]

        # Register buffers for lookups
        self.register_buffer("group_index", torch.tensor(group_table, dtype=torch.long))
        self.register_buffer("period_index", torch.tensor(period_table, dtype=torch.long))
        
        num_tensor = torch.tensor(num_table, dtype=torch.float32)
        # Normalize numerical features
        with torch.no_grad():
            mean = num_tensor.mean(dim=0, keepdim=True)
            std = num_tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
            num_tensor = (num_tensor - mean) / std
        self.register_buffer("num_values", num_tensor)

        # Embeddings
        # Group: 1-18, allow 0 for padding/unknown -> 19 dims
        self.group_embedding = nn.Embedding(20, dim_group, padding_idx=0)
        # Period: 1-7, allow 0 -> 8 dims
        self.period_embedding = nn.Embedding(10, dim_period, padding_idx=0)
        # Tag: 0, 1, 2 -> 3 dims
        self.tag_embedding = nn.Embedding(3, dim_tag)
        
        # Base Z embedding (optional, using smaller dim than output)
        self.z_embedding = nn.Embedding(self.max_Z + 1, 32, padding_idx=0)

        # Input dimension for final project
        # Z(32) + Group(32) + Period(32) + Tag(32) + Num(6) = 134
        in_dim = 32 + dim_group + dim_period + dim_tag + len(self.num_feature_names)
        
        self.project = nn.Sequential(
            nn.Linear(in_dim, max(128, in_dim * 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(128, in_dim * 2), self.out_dim),
            nn.LayerNorm(self.out_dim),
        )

    def forward(self, atomic_numbers: torch.Tensor, tags: torch.Tensor = None) -> torch.Tensor:
        # atomic_numbers: [N]
        idx = atomic_numbers.clamp_min(0).clamp_max(self.max_Z)
        
        # 1. Embeddings
        z_emb = self.z_embedding(idx)          # [N, 32]
        
        g_idx = self.group_index[idx]
        g_emb = self.group_embedding(g_idx)    # [N, 32]
        
        p_idx = self.period_index[idx]
        p_emb = self.period_embedding(p_idx)   # [N, 32]
        
        # 2. Tag Embedding
        if tags is not None:
            # tags usually [N], int (0,1,2)
            t_emb = self.tag_embedding(tags.long()) # [N, 32]
        else:
            # Default to 0? Or maybe create a zeros tensor?
            t_emb = torch.zeros(len(idx), self.tag_embedding.embedding_dim, device=idx.device)
            
        # 3. Numerical Features
        phys = self.num_values[idx]            # [N, 6]
        
        # 4. Concatenate All
        # PhAST style: Z || Group || Period || Tag || Phys
        combined = torch.cat([z_emb, g_emb, p_emb, t_emb, phys], dim=-1)
        
        return self.project(combined)  # [N, out_dim]
