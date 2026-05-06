# core/models/embedding_branch.py

import torch
import torch.nn as nn
from torch_scatter import scatter

from fairchem.core.common.registry import registry
from fairchem.core.models.base import GraphModelMixin
from .equiformer_v2.global_emb import MOFGlobalPropertyEncoder
from .equiformer_v2.atomic_emb import AtomPropertyEncoder


class ResidualMLPBlock(nn.Module):
    """PreNorm Residual MLP block: LN -> Linear(ff*dim) -> GELU -> Dropout -> Linear(dim) -> Dropout + Residual"""

    def __init__(self, dim: int, ff_mult: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(p_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


@registry.register_model("embedding_branch_net")
class EmbeddingBranchNet(nn.Module, GraphModelMixin):
    """
    支网：使用 MOF 全局特征与原子额外特征，通过深层 ResMLP 进行逐原子表征，再回归能量。

    兼容 Trainer 传入的图构建参数（use_pbc/otf_graph/max_neighbors/max_radius/max_num_elements），
    但本模型前向不直接使用这些键。
    """

    def __init__(
        self,
        # Feature config
        use_mof_global_features: bool = True,
        mof_global_dim: int = 32,
        use_atom_extra_features: bool = True,
        atom_feat_dim: int = 16,
        # ResMLP config
        hidden_dim: int = 256,
        num_mlp_layers: int = 3,  # 兼容旧参：若未提供 resmlp_num_blocks，则使用该值近似作为块数
        resmlp_num_blocks: int | None = None,
        resmlp_ff_mult: int = 4,
        resmlp_dropout: float = 0.1,
        # Output scaling
        avg_num_nodes: float = 192.561,
        # Graph/OTF graph related keys (accepted for config compatibility)
        use_pbc: bool | None = None,
        otf_graph: bool | None = None,
        max_neighbors: int | None = None,
        max_radius: float | None = None,
        max_num_elements: int | None = None,
        # Future-proof: accept and ignore unknown keys
        **kwargs,
    ):
        super().__init__()
        self.use_mof_global_features = use_mof_global_features
        self.mof_global_dim = mof_global_dim
        self.use_atom_extra_features = use_atom_extra_features
        self.atom_feat_dim = atom_feat_dim
        self.avg_num_nodes = avg_num_nodes

        # Graph-related flags (stored for reference)
        self.use_pbc = use_pbc
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.max_num_elements = max_num_elements
        self._extra_init_kwargs = kwargs  # keep for debugging if needed

        # --- Feature Encoders ---
        if self.use_mof_global_features:
            self.mof_global_encoder = MOFGlobalPropertyEncoder(
                excel_path='/home/dell/autodl-tmp/lorafair/data/MOF_embedding_train-check580.xlsx',
                global_dim=self.mof_global_dim,
                multi_label_pool='mean',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.mof_global_encoder = None

        if self.use_atom_extra_features:
            self.atom_encoder = AtomPropertyEncoder(max_Z=100, out_dim=self.atom_feat_dim)
        else:
            self.atom_encoder = None

        # --- Core: Deep ResMLP ---
        input_dim = 0
        if self.use_mof_global_features:
            input_dim += self.mof_global_dim
        if self.use_atom_extra_features:
            input_dim += self.atom_feat_dim
        if input_dim == 0:
            raise ValueError("EmbeddingBranchNet requires at least one feature type to be enabled.")

        # 将拼接后的输入升维到隐藏维度
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pre_norm = nn.LayerNorm(hidden_dim)

        # 块数：优先使用 resmlp_num_blocks；否则退回到 max(1, num_mlp_layers)
        num_blocks = resmlp_num_blocks if resmlp_num_blocks is not None else max(1, int(num_mlp_layers))
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, ff_mult=resmlp_ff_mult, p_drop=resmlp_dropout)
            for _ in range(num_blocks)
        ])

        self.energy_head = nn.Linear(hidden_dim, 1)
        self._final_feature_dim = hidden_dim

    def forward(self, data):
        device = data.pos.device
        dtype = data.pos.dtype
        atomic_numbers = data.atomic_numbers.long()
        batch_vec = data.batch

        features_to_concat: list[torch.Tensor] = []

        # 1) MOF 全局特征 → [B, d_g] → 按 batch 展开到 [N, d_g]
        if self.use_mof_global_features and self.mof_global_encoder is not None:
            mof_names_batch = []
            if hasattr(data, 'name') and data.name is not None:
                if isinstance(data.name, list):
                    mof_names_batch = [str(name).split('_')[0] for name in data.name]
                else:  # Fallback for single sample
                    mof_names_batch = [str(data.name).split('_')[0]] * len(data.natoms)

            if len(mof_names_batch) > 0:
                mof_embeddings_list = [
                    self.mof_global_encoder(name) if name else torch.zeros(1, self.mof_global_dim, device=device, dtype=dtype)
                    for name in mof_names_batch
                ]
                mof_global_embedding = torch.cat(mof_embeddings_list, dim=0)
                global_per_node = mof_global_embedding[batch_vec]
                features_to_concat.append(global_per_node)

        # 2) 原子级特征 → [N, d_a]
        if self.use_atom_extra_features and self.atom_encoder is not None:
            atom_feat = self.atom_encoder(atomic_numbers)
            features_to_concat.append(atom_feat)

        if not features_to_concat:
            # Fallback: no features gathered; return zeros with expected shapes
            num_graphs = len(data.natoms)
            energy = torch.zeros(num_graphs, device=device, dtype=dtype)
            node_features = torch.zeros(
                len(atomic_numbers), self._final_feature_dim, device=device, dtype=dtype
            )
            return {"energy": energy, "node_features": node_features}

        # 3) ResMLP 表征
        x = torch.cat(features_to_concat, dim=-1)            # [N, input_dim]
        x = self.input_proj(x)                               # [N, hidden_dim]
        x = self.pre_norm(x)
        for blk in self.blocks:
            x = blk(x)                                       # [N, hidden_dim]
        node_features = x

        # 4) 逐原子能量 & 聚合
        node_energy = self.energy_head(node_features)        # [N, 1]
        energy = scatter(node_energy, batch_vec, dim=0, reduce='sum').squeeze(-1)  # [B]
        energy = energy / self.avg_num_nodes

        return {"energy": energy, "node_features": node_features}
