
from __future__ import annotations
import os
import re
import sys
import torch
import yaml
import math
import logging
import contextlib
import torch.nn as nn
import loralib as lora
import torch.nn.functional as F
from typing import Optional
from fairchem.core.common import gp_utils
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import GraphModelMixin
from fairchem.core.models.scn.smearing import GaussianSmearing
from fairchem.core.modules.scaling.compat import load_scales_compat

with contextlib.suppress(ImportError):
    pass
from .edge_rot_mat import init_edge_rot_mat
from .gaussian_rbf import GaussianRadialBasisLayer
from .input_block import EdgeDegreeEmbedding
from .layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .module_list import ModuleListInfo
from .radial_function import RadialFunction
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
    SO3_Rotation,
)
from .transformer_block import (
    FeedForwardNetwork,
    SO2EquivariantGraphAttention,
    TransBlockV2,
)
from .global_emb import MOFGlobalPropertyEncoder
from .atomic_emb import AtomPropertyEncoder
# V2 编码器（三模态 + 物理特征）
try:
    from .global_emb_v2 import MOFGlobalEncoderV2
except Exception:
    MOFGlobalEncoderV2 = None
try:
    from .atomic_emb_v2 import AtomPropertyEncoderV2
except Exception:
    AtomPropertyEncoderV2 = None
# nm (condition) 编码器，复用 eSCN 实现
try:
    from fairchem.core.models.escn.escn import ConditionEncoder
except Exception:
    ConditionEncoder = None

# Statistics of IS2RE 100K
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773  # IS2RE: 100k, max_radius = 5, max_neighbors = 100





# class LoRALinear(nn.Module):
#     """LoRA线性层实现"""
#     def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 16.0, dropout: float = 0.1):
#         super().__init__()
#         self.original_layer = original_layer
#         self.rank = rank
#         self.alpha = alpha
#         
#         # 冻结原始权重
#         for param in self.original_layer.parameters():
#             param.requires_grad = False
#         
#         # LoRA参数
#         self.lora_A = nn.Parameter(torch.randn(rank, original_layer.in_features) * 0.01)
#         self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
#         self.dropout = nn.Dropout(dropout)
#         
#         # 缩放因子
#         self.scaling = alpha / rank
#     
#     def forward(self, x):
#         # 原始输出
#         original_output = self.original_layer(x)
#         
#         # LoRA输出
#         lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
#         
#         return original_output + lora_output * self.scaling

class LoRALinear(nn.Module):
    """
    包装原始 nn.Linear 的 LoRA 实现：
    - 冻结原始权重
    - 仅训练低秩 A,B
    - 暴露与 nn.Linear 接口兼容的属性：in_features / out_features / weight / bias
    - 支持 merge() 将 LoRA 权重合并回原层（推理时可选）
    """
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_scale: float = 0.01,
        enable_lora: bool = True,
    ):
        super().__init__()
        assert isinstance(original_layer, nn.Linear), "original_layer 必须是 nn.Linear"

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out  # 某些框架里权重可能转置存储
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 复制接口属性（下游会直接访问）
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # 冻结原始权重
        for p in self.original_layer.parameters():
            p.requires_grad = False

        weight_dtype = self.original_layer.weight.dtype
        weight_device = self.original_layer.weight.device

        if enable_lora and rank > 0:
            # A: (rank, in_features), B: (out_features, rank)
            self.lora_A = nn.Parameter(
                torch.randn(rank, self.in_features, device=weight_device, dtype=weight_dtype) * init_scale
            )
            self.lora_B = nn.Parameter(
                torch.zeros(self.out_features, rank, device=weight_device, dtype=weight_dtype)
            )
        else:
            # 关闭 LoRA
            self.lora_A = None
            self.lora_B = None

        self.merged = False  # 标记是否已经 merge

    # 通过 property 暴露 weight / bias，让外部像用 nn.Linear 一样访问
    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged or (self.lora_A is None) or (not self.enable_lora):
            # 直接走原层
            return self.original_layer(x)

        # 原始输出
        result = self.original_layer(x)

        # LoRA 增量: x @ A^T  -> (batch, rank)
        lora_intermediate = self.dropout(x) @ self.lora_A.t()  # [B, rank]
        # (batch, rank) @ B^T -> (batch, out_features)
        lora_update = lora_intermediate @ self.lora_B.t()
        result = result + lora_update * self.scaling
        return result

    @torch.no_grad()
    def merge(self):
        """
        将 LoRA 权重合并到原始权重 (W += B @ A * scaling)，之后可以删除 A,B 节省计算。
        适合推理时调用；训练中不要 merge。
        """
        if self.merged:
            return
        if self.lora_A is None:
            self.merged = True
            return
        delta_w = (self.lora_B @ self.lora_A) * self.scaling  # (out_features, in_features)
        self.original_layer.weight.data += delta_w.to(self.original_layer.weight.dtype)
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        """
        如果 merge 后又想继续微调，需要 unmerge。
        """
        if not self.merged or self.lora_A is None:
            return
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.original_layer.weight.data -= delta_w.to(self.original_layer.weight.dtype)
        self.merged = False

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling}, "
            f"enable_lora={self.enable_lora}, merged={self.merged}"
        )


class LoRATRFLinear(nn.Module):
    """
    An enhanced LoRA layer that incorporates Task-Relevant Feature enhancement.
    This layer applies two parallel updates to the original linear layer's output:
    1. A standard LoRA update (low-rank weight modification).
    2. A LoRATRF update (task-aware representation editing).
    """
    def __init__(
        self,
        original_layer: nn.Linear,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        trf_rank: int = 8,  # Rank for the Task-Relevant Feature transformation
        trf_alpha: float = 1.0, # Scaling factor for the TRF update
    ):
        super().__init__()
        assert isinstance(original_layer, nn.Linear), "original_layer must be nn.Linear"

        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # --- Standard LoRA Path ---
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # --- Task-Relevant Feature (TRF) Path ---
        self.trf_rank = trf_rank
        self.trf_alpha = trf_alpha

        # Freeze original weights
        for p in self.original_layer.parameters():
            p.requires_grad = False

        weight_dtype = self.original_layer.weight.dtype
        weight_device = self.original_layer.weight.device

        # LoRA parameters
        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.randn(lora_rank, self.in_features, device=weight_device, dtype=weight_dtype) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, lora_rank, device=weight_device, dtype=weight_dtype))
        else:
            self.lora_A = None
            self.lora_B = None

        # TRF parameters
        if self.trf_rank > 0:
            # Learnable, layer-specific task vector
            self.trf_task_vector = nn.Parameter(torch.randn(1, self.out_features, device=weight_device, dtype=weight_dtype))
            # Low-rank transformation matrices
            self.trf_C = nn.Linear(self.out_features, self.trf_rank, bias=False, device=weight_device, dtype=weight_dtype)
            self.trf_D = nn.Linear(self.trf_rank, self.out_features, bias=False, device=weight_device, dtype=weight_dtype)
            # Initialize D as zero for stable initial training
            nn.init.zeros_(self.trf_D.weight)
        else:
            self.trf_task_vector = None
            self.trf_C = None
            self.trf_D = None

    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Base path
        h = self.original_layer(x)
        result = h

        # 2. Standard LoRA path (operates on input x)
        if self.lora_A is not None:
            lora_intermediate = self.lora_dropout(x) @ self.lora_A.t()
            lora_update = lora_intermediate @ self.lora_B.t()
            result = result + lora_update * self.lora_scaling

        # 3. TRF path (operates on output h)
        if self.trf_task_vector is not None:
            # Task-aware filtering
            similarity = F.cosine_similarity(h, self.trf_task_vector, dim=-1).unsqueeze(-1)
            similarity = torch.clamp(similarity, 0.0, 1.0)
            filtered_h = h * similarity

            # Low-rank transformation
            trf_update = self.trf_D(self.trf_C(filtered_h))
            result = result + trf_update * self.trf_alpha
            
        return result

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"lora_rank={self.lora_rank}, lora_alpha={self.lora_alpha}, "
            f"trf_rank={self.trf_rank}, trf_alpha={self.trf_alpha}"
        )
        




@registry.register_model("equiformer_v2_lora")
class EquiformerV2LoRA(nn.Module, GraphModelMixin):
    """
    等变性保证的EquiformerV2 LoRA微调版本
    """
    
    def __init__(
        self,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        regress_forces: bool = True,
        otf_graph: bool = True,
        max_neighbors: int = 500,
        max_radius: float = 5.0,
        max_num_elements: int = 90,
        num_layers: int = 12,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 128,
        num_heads: int = 8,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 512,
        norm_type: str = "rms_norm_sh",
        lmax_list: list[int] | None = None,
        mmax_list: list[int] | None = None,
        grid_resolution: int | None = None,
        num_sphere_samples: int = 128,
        edge_channels: int = 128,
        use_atom_edge_embedding: bool = True,
        share_atom_edge_embedding: bool = False,
        use_m_share_rad: bool = False,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        attn_activation: str = "scaled_silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "scaled_silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = True,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        proj_drop: float = 0.0,
        weight_init: str = "normal",
        enforce_max_neighbors_strictly: bool = True,
        avg_num_nodes: float | None = None,
        avg_degree: float | None = None,
        use_energy_lin_ref: bool | None = False,
        load_energy_lin_ref: bool | None = False,
        
        # LoRA特定参数
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 64.0,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
        # Enhanced LoRA (LoRATRF) parameters
        use_enhanced_lora: bool = False,
        trf_rank: int = 8,
        trf_alpha: float = 1.0,
        # 全局嵌入有关
        use_mof_global_features: bool = False,
        mof_global_dim: int = 32,
        # V2 版 global encoder 选项（"v1" = BiLSTM，"v2" = SMI-TED + SELFIES-TED + MHG 三模态）
        mof_global_encoder_type: str = "v1",
        mof_global_excel_path: Optional[str] = None,
        mof_smi_ted_folder: Optional[str] = None,
        mof_selfies_ted_path: Optional[str] = None,
        mof_mhg_path: Optional[str] = None,
        mof_smiles_cache_path: Optional[str] = None,
        mof_global_pretrained_device: Optional[str] = None,

        # 新增：nm(condition) 编码器（与 eSCN/PaiNN 一致）
        base_nm_dim: int = 2,
        condition_hidden_dim: int = 64,
        nm_max_count: float = 15.0,
        use_nm_features: bool = True,   # 是否加入 nm condition（默认开）

        # 新增：融合/注入模式（默认 film）
        feature_fusion: str = "film",   # film | gating | add | concat | cross_attn | loratrf
        # 全局 FiLM 注入层（0 = layer 0 前，N > 0 = layer N 前；eSCN 风格中层注入）
        # 默认 4 = 8 层模型的中点
        global_inject_layer: int = 4,
        # Stage 2 硬冻结前 N 个 block 的 LoRA（对齐 eSCN 的 freeze_first_n_layers 语义）
        # 默认 0 = 不冻结任何 LoRA（stage1 行为）
        # stage2 设为 4 = 冻结前 4 个 block 的 LoRA，只训练后 4 个 block 的 LoRA
        freeze_first_n_lora_blocks: int = 0,
        # 适配器/注入的配置
        adapter_hidden_dim: int | None = None,
        adapter_dropout: float = 0.1,
        adapter_zero_init: bool = True,
        apply_context_layernorm: bool = True,
        # GemNet 风格：仅对 global 分支启用归一化（global encoder 输出 + FiLM 隐藏层）
        # 默认 False，与此前配置完全兼容
        global_layer_norm: bool = False,

        ## pretraine path
        pretrained_model_path: Optional[str] = '/home/dell/autodl-tmp/lorafair/fairchem/lora_model/Equiformer_V2_Direct.pt',
        # 原子级特征
        use_atom_extra_features: bool = False,
        atom_feat_out_dim: int = 128,
        # V2 版 atom encoder 选项（"v1" = AtomPropertyEncoder, "v2" = AtomPropertyEncoderV2）
        atom_encoder_type: str = "v1",
        # LoRATRF specific parameters
        loratrf_rank: int = 8,
        # Train all parameters related to scalar (l=0) features
        train_scalar_path: bool = False,
        # 仅解冻输出头（能量头）
        train_energy_head: bool = False,
        # Embedding branch network configuration
        use_embedding_branch: bool = False,
        embedding_branch_path: Optional[str] = None,
        freeze_embedding_branch: bool = True,
        embedding_branch_hidden_dim: int = 256, # Must match the hidden_dim of the trained branch net
    ):
        # 重写
        logging.warning(
            "equiformer_v2 (EquiformerV2) class is deprecated in favor of equiformer_v2_backbone_and_heads  (EquiformerV2BackboneAndHeads)"
        )
        if mmax_list is None:
            mmax_list = [2]
        super().__init__()
        if lmax_list is None:
            lmax_list = [6]

        import sys

        if "e3nn" not in sys.modules:
            logging.error("You need to install e3nn==0.4.4 to use EquiformerV2.")
            raise ImportError

        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis
        # expose configurable feature dims
        self.atom_feat_out_dim = atom_feat_out_dim
        self.mof_global_dim = mof_global_dim

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.avg_num_nodes = avg_num_nodes or _AVG_NUM_NODES
        self.avg_degree = avg_degree or _AVG_DEGREE

        self.use_energy_lin_ref = use_energy_lin_ref
        self.load_energy_lin_ref = load_energy_lin_ref
        assert not (
            self.use_energy_lin_ref and not self.load_energy_lin_ref
        ), "You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine."

        self.weight_init = weight_init
        assert self.weight_init in ["normal", "uniform"]

        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly

        # --- Store method-switching parameters first ---

        self.use_enhanced_lora = use_enhanced_lora
        self.train_scalar_path = train_scalar_path


        self.device = "cpu"  # torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions: int = len(self.lmax_list)
        self.sphere_channels_all: int = self.num_resolutions * self.sphere_channels

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels_all
        )

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            "gaussian",
        ]
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                self.num_distance_basis,
                2.0,
            )
            # 诊断输出：确认 RBF 基数
            try:
                print(f"[Diag] distance_expansion.num_output={int(self.distance_expansion.num_output)}; config.num_distance_basis={self.num_distance_basis}")
            except Exception:
                pass
            # self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError

        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [
            self.edge_channels
        ] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo(
            f"({max(self.lmax_list)}, {max(self.lmax_list)})"
        )
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=self.grid_resolution,
                        normalization="component",
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding 将边的标量特征（距离、原子类型）转换为球谐函数嵌入
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list, # 600,128,128
            self.block_use_atom_edge_embedding,
            rescale_factor=self.avg_degree,
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop,
            )
            self.blocks.append(block)

        # Output blocks for energy and forces
        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list),
            num_channels=self.sphere_channels,
        )
        self.energy_block = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels,
            1,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
        )
        if self.regress_forces:
            self.force_block = SO2EquivariantGraphAttention(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0,
            )

        if self.load_energy_lin_ref:
            self.energy_lin_ref = nn.Parameter(
                torch.zeros(self.max_num_elements),
                requires_grad=False,
            )

        # 🔧 修复：在加载预训练权重之前，先创建所有额外模块
        # 这样预训练权重加载时才能覆盖这些模块的初始化
        #全局嵌入相关
        self.use_mof_global_features = use_mof_global_features
        self.fusion_method = feature_fusion.lower()
        self.mof_global_encoder_type = mof_global_encoder_type.lower().strip()
        self.atom_encoder_type = atom_encoder_type.lower().strip()
        self.mof_global_dim = mof_global_dim

        # 全局特征编码（V1 = BiLSTM / V2 = 三模态预训练模型）
        self.mof_global_encoder = None
        if self.use_mof_global_features:
            _dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            if self.mof_global_encoder_type == "v2":
                if MOFGlobalEncoderV2 is None:
                    logging.warning("[EqV2-LoRA] MOFGlobalEncoderV2 不可用，回退到 V1")
                    self.mof_global_encoder_type = "v1"
                else:
                    try:
                        self.mof_global_encoder = MOFGlobalEncoderV2(
                            excel_path=mof_global_excel_path,
                            smi_ted_folder=mof_smi_ted_folder,
                            selfies_ted_path=mof_selfies_ted_path,
                            mhg_path=mof_mhg_path,
                            global_dim=mof_global_dim,
                            device=_dev,
                            pretrained_device=mof_global_pretrained_device,
                            cache_path=mof_smiles_cache_path,
                        )
                        logging.info("[EqV2-LoRA] MOFGlobalEncoderV2 (三模态) 初始化成功")
                    except Exception as e:
                        logging.warning(f"[EqV2-LoRA] MOFGlobalEncoderV2 init failed: {e}, 回退到 V1")
                        self.mof_global_encoder_type = "v1"
            if self.mof_global_encoder_type == "v1" and self.mof_global_encoder is None:
                self.mof_global_encoder = MOFGlobalPropertyEncoder(
                    excel_path=mof_global_excel_path or
                               '/home/dell/autodl-tmp/lorafair/data/MOF_embedding_train-check580.xlsx',
                    global_dim=mof_global_dim,
                    multi_label_pool='mean',
                    device=_dev,
                )
                logging.info("[EqV2-LoRA] MOFGlobalPropertyEncoder (BiLSTM V1) 初始化成功")

        # 原子级特征编码（V1 = AtomPropertyEncoder / V2 = AtomPropertyEncoderV2）
        self.use_atom_extra_features = use_atom_extra_features
        self.atom_encoder = None
        if self.use_atom_extra_features:
            if self.atom_encoder_type == "v2":
                if AtomPropertyEncoderV2 is None:
                    logging.warning("[EqV2-LoRA] AtomPropertyEncoderV2 不可用，回退到 V1")
                    self.atom_encoder_type = "v1"
                else:
                    self.atom_encoder = AtomPropertyEncoderV2(
                        max_Z=self.max_num_elements, out_dim=self.atom_feat_out_dim
                    )
                    logging.info(f"[EqV2-LoRA] AtomPropertyEncoderV2 init OK, out_dim={self.atom_feat_out_dim}")
            if self.atom_encoder_type == "v1" and self.atom_encoder is None:
                self.atom_encoder = AtomPropertyEncoder(
                    max_Z=self.max_num_elements, out_dim=self.atom_feat_out_dim
                )

        # nm(condition) 编码器
        self.use_nm_features = bool(use_nm_features)
        self.base_nm_dim = int(base_nm_dim)
        self.nm_max_count = float(nm_max_count)
        self.nm_encoded_dim = 0
        self.condition_encoder = None
        if self.use_nm_features:
            if ConditionEncoder is None:
                logging.warning("[EqV2-LoRA] ConditionEncoder 不可用，禁用 nm 注入")
                self.use_nm_features = False
            else:
                self.condition_encoder = ConditionEncoder(
                    input_dim=base_nm_dim,
                    num_gaussians=20,
                    out_dim=condition_hidden_dim,
                    hidden_dim=adapter_hidden_dim or 128,
                    dropout=adapter_dropout,
                    nm_max_count=nm_max_count,
                )
                self.nm_encoded_dim = condition_hidden_dim
                logging.info(f"[EqV2-LoRA] ConditionEncoder init OK, nm_encoded_dim={condition_hidden_dim}")

        # --- Embedding Branch Network ---（保持原逻辑）
        self.use_embedding_branch = use_embedding_branch
        self.embedding_branch_path = embedding_branch_path
        self.freeze_embedding_branch = freeze_embedding_branch
        
        if self.use_embedding_branch:
            # 为了避免循环导入，这里本文件下方定义的 EmbeddingBranchNet 需存在于路径
            from fairchem.core.models.embedding_branch import EmbeddingBranchNet
            self.embedding_branch_net = EmbeddingBranchNet(
                use_mof_global_features=self.use_mof_global_features,
                mof_global_dim=mof_global_dim,
                use_atom_extra_features=self.use_atom_extra_features,
                atom_feat_dim=self.atom_feat_out_dim,
                hidden_dim=embedding_branch_hidden_dim
            )
            self.embedding_branch_proj = nn.Linear(embedding_branch_hidden_dim, self.sphere_channels)
            if self.embedding_branch_path:
                logging.info(f"Loading pretrained embedding branch from: {self.embedding_branch_path}")
                branch_ckpt = torch.load(self.embedding_branch_path, map_location='cpu')
                branch_state_dict = branch_ckpt.get('state_dict', branch_ckpt)
                cleaned_branch_state_dict = {k.replace('module.', ''): v for k, v in branch_state_dict.items()}
                self.embedding_branch_net.load_state_dict(cleaned_branch_state_dict)
                logging.info("Successfully loaded pretrained embedding branch.")
            if self.freeze_embedding_branch:
                logging.info("Freezing embedding branch network parameters.")
                for param in self.embedding_branch_net.parameters():
                    param.requires_grad = False
        else:
            self.embedding_branch_net = None
        
        # --- 融合/注入模块构建 ---
        # eSCN/PaiNN 模式：context 只包含 nm + atom，global 走独立 FiLM 分支
        # 这样 stage1 (nm+atom) → stage2 (+global) 时，context_ln / film_gamma / film_beta
        # 维度完全一致，checkpoint 能干净加载；global 部分 zero-init → stage2 初始 = stage1。
        self.context_input_dim = 0
        if self.use_nm_features and self.nm_encoded_dim > 0:
            self.context_input_dim += self.nm_encoded_dim
        if self.use_atom_extra_features:
            self.context_input_dim += self.atom_feat_out_dim
        # 注意：mof_global_dim 不再进入 context_input_dim（只 film 路径使用独立 global FiLM）

        # 全局 FiLM 注入层（eSCN 风格）
        self.global_inject_layer = int(global_inject_layer)
        # 硬冻结前 N 个 block 的 LoRA（stage 2 对齐 eSCN 的 freeze_first_n_layers）
        self.freeze_first_n_lora_blocks = int(freeze_first_n_lora_blocks)

        self.apply_context_layernorm = apply_context_layernorm
        if self.context_input_dim > 0 and self.apply_context_layernorm:
            self.context_ln = nn.LayerNorm(self.context_input_dim)
        else:
            self.context_ln = None

        # 适配器隐藏维度
        self.adapter_hidden_dim = adapter_hidden_dim or self.sphere_channels
        self.adapter_dropout = adapter_dropout
        self.adapter_zero_init = adapter_zero_init

        # concat 路径（保留原实现，同时把 nm 也 concat 进去）
        fusion_input_dim = self.sphere_channels
        if self.use_nm_features and self.nm_encoded_dim > 0:
            fusion_input_dim += self.nm_encoded_dim
        if self.use_mof_global_features:
            fusion_input_dim += mof_global_dim
        if self.use_atom_extra_features:
            fusion_input_dim += self.atom_feat_out_dim
        if self.fusion_method == "concat":
            if self.context_input_dim > 0:
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(fusion_input_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.LayerNorm(256),
                    nn.Linear(256, self.sphere_channels),
                    nn.LayerNorm(self.sphere_channels),
                )
            else:
                # 无上下文特征时不创建，以免随机初始化影响初始输出
                self.fusion_mlp = None
        else:
            self.fusion_mlp = None

        # Cross-Attention 路径（仅当存在上下文特征时才构建）
        if self.fusion_method == "cross_attn" and self.context_input_dim > 0:
            if self.use_mof_global_features:
                self.global_proj = nn.Linear(mof_global_dim, self.sphere_channels)
            else:
                self.global_proj = None
            if self.use_atom_extra_features:
                self.atom_proj = nn.Linear(self.atom_feat_out_dim, self.sphere_channels)
            else:
                self.atom_proj = None
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.sphere_channels,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.fusion_proj = nn.Sequential(
                nn.Linear(self.sphere_channels, self.sphere_channels),
                nn.LayerNorm(self.sphere_channels),
                nn.Dropout(0.1)
            )
        else:
            # 无上下文特征或非 cross_attn 模式下，不创建这些模块，避免随机初始化与多余可训练参数
            self.global_proj = None
            self.atom_proj = None
            self.cross_attn = None
            self.fusion_proj = None

        # LoRATRF 路径（保持原逻辑）
        if self.fusion_method == "loratrf":
            context_input_dim = self.context_input_dim
            if context_input_dim > 0:
                self.loratrf_context_proj = nn.Linear(context_input_dim, self.sphere_channels)
                self.loratrf_task_vector = None
            else:
                self.loratrf_context_proj = None
                self.loratrf_task_vector = nn.Parameter(torch.randn(1, self.sphere_channels))
            self.loratrf_transform_C = nn.Linear(self.sphere_channels, loratrf_rank, bias=False)
            self.loratrf_transform_D = nn.Linear(loratrf_rank, self.sphere_channels, bias=False)
            nn.init.zeros_(self.loratrf_transform_D.weight)
        else:
            self.loratrf_context_proj = None
            self.loratrf_task_vector = None
            self.loratrf_transform_C = None
            self.loratrf_transform_D = None

        # 新增：FiLM / Gating / Add 注入路径（默认开启 film）
        def _zero_last_linear(module: nn.Sequential | nn.Module):
            # 将序列中的最后一个 Linear 权重、偏置置零
            if isinstance(module, nn.Sequential):
                for m in reversed(module):
                    if isinstance(m, nn.Linear):
                        nn.init.zeros_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                        break
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self.fusion_method in ["film", "gating", "add"] and self.context_input_dim > 0:
            # 统一的上下文 → 隐层投影
            if self.fusion_method == "film":
                self.film_gamma = nn.Sequential(
                    nn.Linear(self.context_input_dim, self.adapter_hidden_dim),
                    nn.LayerNorm(self.adapter_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.adapter_dropout),
                    nn.Linear(self.adapter_hidden_dim, self.sphere_channels),
                )
                self.film_beta = nn.Sequential(
                    nn.Linear(self.context_input_dim, self.adapter_hidden_dim),
                    nn.LayerNorm(self.adapter_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.adapter_dropout),
                    nn.Linear(self.adapter_hidden_dim, self.sphere_channels),
                )
                if self.adapter_zero_init:
                    _zero_last_linear(self.film_gamma)
                    _zero_last_linear(self.film_beta)
                self.gate_mlp = None
                self.add_mlp = None
            elif self.fusion_method == "gating":
                self.film_gamma = None
                self.film_beta = None
                self.add_mlp = None
                self.gate_mlp = nn.Sequential(
                    nn.Linear(self.context_input_dim, self.adapter_hidden_dim),
                    nn.LayerNorm(self.adapter_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.adapter_dropout),
                    nn.Linear(self.adapter_hidden_dim, self.sphere_channels),
                    nn.Tanh(),
                )
                if self.adapter_zero_init:
                    for m in reversed(self.gate_mlp):
                        if isinstance(m, nn.Linear):
                            nn.init.zeros_(m.weight)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)
                            break
            else:  # add
                self.film_gamma = None
                self.film_beta = None
                self.gate_mlp = None
                self.add_mlp = nn.Sequential(
                    nn.Linear(self.context_input_dim, self.adapter_hidden_dim),
                    nn.LayerNorm(self.adapter_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.adapter_dropout),
                    nn.Linear(self.adapter_hidden_dim, self.sphere_channels),
                )
                if self.adapter_zero_init:
                    _zero_last_linear(self.add_mlp)

            # ── Global FiLM 独立分支（仅 film 模式 + global 启用时创建）──────────
            # 默认：严格对齐 eSCN 的简化结构（Linear → GELU → Linear），无 LayerNorm/Dropout。
            # 当 global_layer_norm=True 时：采用 GemNet 风格的 global 归一化
            #   - 注入前对 global_per_node 过一层 LayerNorm(mof_global_dim)
            #   - FiLM MLP 隐藏层 (Linear 后) 再加 LayerNorm(adapter_hidden_dim)
            self.global_gamma_mlp = None
            self.global_beta_mlp = None
            self.global_input_norm = None
            self.global_layer_norm = bool(global_layer_norm)
            if self.fusion_method == "film" and self.use_mof_global_features:
                if self.global_layer_norm:
                    self.global_input_norm = nn.LayerNorm(mof_global_dim)
                self.global_gamma_mlp = nn.Sequential(
                    nn.Linear(mof_global_dim, self.adapter_hidden_dim),
                    nn.LayerNorm(self.adapter_hidden_dim) if self.global_layer_norm else nn.Identity(),
                    nn.GELU(),
                    nn.Linear(self.adapter_hidden_dim, self.sphere_channels),
                )
                self.global_beta_mlp = nn.Sequential(
                    nn.Linear(mof_global_dim, self.adapter_hidden_dim),
                    nn.LayerNorm(self.adapter_hidden_dim) if self.global_layer_norm else nn.Identity(),
                    nn.GELU(),
                    nn.Linear(self.adapter_hidden_dim, self.sphere_channels),
                )
                if self.adapter_zero_init:
                    _zero_last_linear(self.global_gamma_mlp)
                    _zero_last_linear(self.global_beta_mlp)
                if self.global_layer_norm:
                    logging.info("[EqV2-LoRA] global_gamma_mlp / global_beta_mlp (GemNet 风格，global_layer_norm=True) init OK, zero-init")
                else:
                    logging.info("[EqV2-LoRA] global_gamma_mlp / global_beta_mlp (简化版 eSCN 对齐) init OK, zero-init")
        else:
            self.film_gamma = None
            self.film_beta = None
            self.gate_mlp = None
            self.add_mlp = None
            self.global_gamma_mlp = None
            self.global_beta_mlp = None
            self.global_input_norm = None
            self.global_layer_norm = False

        # LoRA配置
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # --- Store method-switching parameters first for access in helper methods ---
        self.use_enhanced_lora = use_enhanced_lora
        self.train_scalar_path = train_scalar_path
        self.train_energy_head = train_energy_head
        self.fusion_method = feature_fusion.lower()
        self.trf_rank = trf_rank
        self.trf_alpha = trf_alpha

        
        



        
        # LoRA目标模块
        if lora_target_modules is None:
            lora_target_modules = [
                'ga.so2_conv_2.fc_m0',      # 针对 ga 模块中的 fc_m0
                'ffn.scalar_mlp' # 针对 ffn 模块中的 scalar_mlp
            ]
        self.lora_target_modules = lora_target_modules
        self.lora_config = {
            'rank': self.lora_rank,
            "alpha" : self.lora_alpha,
            "dropout" : self.lora_dropout,
            "target_modules" : self.lora_target_modules}
        self.lora_modules = {}
        
        # 🔧 修复：所有额外模块创建完成后，现在加载预训练权重
        # 这样预训练权重才能覆盖所有模块的初始化
        if pretrained_model_path is not None:
            self._load_pretrained_weights(pretrained_model_path)
        else:
            # 只有在没有预训练权重时才进行初始化
            self.apply(self._init_weights)
            self.apply(self._uniform_init_rad_func_linear_weights)
        
        # 应用LoRA
        if self.use_lora:
            self._apply_lora()

        # Dynamically set trainable modules based on the current configuration
        self._configure_trainable_modules()
        # Freeze parameters accordingly
        self._freeze_base_model()

        # ── Stage 2 风格：硬冻结前 N 个 block 的 LoRA（对齐 eSCN freeze_first_n_layers） ──
        if self.use_lora and self.freeze_first_n_lora_blocks > 0:
            self._freeze_first_lora_blocks(self.freeze_first_n_lora_blocks)





    def _configure_trainable_modules(self):
        """Dynamically constructs the list of trainable modules based on the config."""
        self.trainable_modules = [
            'ga.so2_conv_2.fc_m0',
            'ffn.scalar_mlp.0',
        ]

        if self.use_mof_global_features:
            self.trainable_modules.append('mof_global_encoder')
        if self.use_atom_extra_features:
            self.trainable_modules.append('atom_encoder')

        if self.fusion_method == "concat" and self.context_input_dim > 0:
            self.trainable_modules.append('fusion_mlp')
        elif self.fusion_method == "cross_attn" and self.context_input_dim > 0:
            self.trainable_modules.extend(['global_proj', 'atom_proj', 'cross_attn', 'fusion_proj'])
        elif self.fusion_method == "loratrf" and self.context_input_dim > 0:
            self.trainable_modules.extend([
                'loratrf_context_proj',
                'loratrf_task_vector',
                'loratrf_transform_C',
                'loratrf_transform_D',
            ])
        elif self.fusion_method in ["film", "gating", "add"] and self.context_input_dim > 0:
            # 适配器模块
            if self.film_gamma is not None:
                self.trainable_modules.append('film_gamma')
            if self.film_beta is not None:
                self.trainable_modules.append('film_beta')
            if self.gate_mlp is not None:
                self.trainable_modules.append('gate_mlp')
            if self.add_mlp is not None:
                self.trainable_modules.append('add_mlp')
            if self.context_ln is not None:
                self.trainable_modules.append('context_ln')
            # 独立 global FiLM 分支
            if getattr(self, 'global_gamma_mlp', None) is not None:
                self.trainable_modules.append('global_gamma_mlp')
            if getattr(self, 'global_beta_mlp', None) is not None:
                self.trainable_modules.append('global_beta_mlp')
            if getattr(self, 'global_input_norm', None) is not None:
                self.trainable_modules.append('global_input_norm')
        # condition_encoder（nm 注入）始终可训练（如果启用）
        if getattr(self, 'use_nm_features', False) and getattr(self, 'condition_encoder', None) is not None:
            self.trainable_modules.append('condition_encoder')

        if self.use_embedding_branch and not self.freeze_embedding_branch:
            self.trainable_modules.append('embedding_branch_net')
        if self.use_embedding_branch:
            self.trainable_modules.append('embedding_branch_proj')

        # 仅解冻输出能量头
        if getattr(self, 'train_energy_head', False):
            self.trainable_modules.append('energy_block')

        if self.train_scalar_path:
            scalar_modules = [
                'sphere_embedding',
                'edge_degree_embedding',
                'ga.so2_conv_1.fc_m0',
                'ga.alpha_func',
                'ga.value_func',
                'ffn.scalar_mlp',
                'norm',
                'energy_block',
            ]
            current_trainable = set(self.trainable_modules)
            current_trainable.update(scalar_modules)
            self.trainable_modules = list(current_trainable)

    def _load_pretrained_weights(self, pretrained_model_path: str):
        """
        加载预训练权重
        """
        import os
        print(f"Loading pretrained weights from: {pretrained_model_path}")
        if not os.path.exists(pretrained_model_path):
            print(f"❌ ERROR: File not found: {pretrained_model_path}")
            return
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            print("✅ Detected checkpoint format, extracting 'state_dict'...")
            pretrained_state_dict = checkpoint['state_dict']
        else:
            print("✅ Detected raw state_dict format...")
            pretrained_state_dict = checkpoint
        cleaned_state_dict = {}
        for key, value in pretrained_state_dict.items():
            if key.startswith('module.module.'):
                new_key = key[14:]
                cleaned_state_dict[new_key] = value
            elif key.startswith('module.'):
                new_key = key[7:]
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value
        pretrained_state_dict = cleaned_state_dict
        current_state_dict = self.state_dict()
        print(f"Pretrained model keys: {len(pretrained_state_dict)}")
        print(f"Current model keys: {len(current_state_dict)}")
        # 根据是否启用额外模块，决定严格加载策略：
        # - 若未启用额外分支/编码器：strict=True（完全对齐）
        # - 若启用了（如 film/gating/add/cross_attn/loratrf 或 MOF/atom 编码器）：
        #   这些模块在 ckpt 中不存在，允许 missing keys → strict=False，并打印清单
        extra_modules_enabled = (
            self.use_mof_global_features or self.use_atom_extra_features or
            (self.fusion_method in ["film", "gating", "add", "cross_attn", "loratrf"]) or
            self.use_embedding_branch  # 预留
        )
        if not extra_modules_enabled:
            try:
                self.load_state_dict(pretrained_state_dict, strict=True)
            except RuntimeError as e:
                print("\n❌ load_state_dict(strict=True) 失败，权重与模型不完全匹配：")
                print(str(e))
                print("\n提示：若为形状不匹配，请检查 RBF 基数、模块选择与预训练配置是否一致（num_distance_basis 应为 600）。")
                raise
            else:
                print("✅ load_state_dict(strict=True) 成功：所有权重完全对齐。")
                print(f"[Diag] distance_expansion.num_output={int(getattr(self.distance_expansion,'num_output',-1))}; config.num_distance_basis={self.num_distance_basis}")
        else:
            # 允许缺失额外模块的权重（这些模块将随机初始化并参与微调）
            load_result = self.load_state_dict(pretrained_state_dict, strict=False)
            missing = load_result.missing_keys
            unexpected = load_result.unexpected_keys
            print("⚠️  启用了额外模块，使用 strict=False 加载预训练权重。")
            print(f"   Missing keys: {len(missing)}")
            if len(missing) > 0:
                # 仅打印与额外模块相关的前若干项，避免刷屏
                preview = [k for k in missing if any(p in k for p in [
                    'mof_global_encoder', 'atom_encoder', 'context_ln', 'film_', 'gate_mlp', 'add_mlp', 'loratrf_'
                ])]
                if preview:
                    print("   示例缺失(额外模块)：", preview[:10], ("..." if len(preview) > 10 else ""))
            print(f"   Unexpected keys: {len(unexpected)}")

    def _init_gp_partitions(
        self,
        atomic_numbers_full,
        data_batch_full,
        edge_index,
        edge_distance,
        edge_distance_vec,
    ):
        node_partition = gp_utils.scatter_to_model_parallel_region(
            torch.arange(len(atomic_numbers_full)).to(self.device)
        )
        edge_partition = torch.where(
            torch.logical_and(
                edge_index[1] >= node_partition.min(),
                edge_index[1] <= node_partition.max(),
            )
        )[0]
        edge_index = edge_index[:, edge_partition]
        edge_distance = edge_distance[edge_partition]
        edge_distance_vec = edge_distance_vec[edge_partition]
        atomic_numbers = atomic_numbers_full[node_partition]
        data_batch = data_batch_full[node_partition]
        node_offset = node_partition.min().item()
        return (
            atomic_numbers,
            data_batch,
            node_offset,
            edge_index,
            edge_distance,
            edge_distance_vec,
        )
    def _apply_lora(self):
        if getattr(self, "_lora_applied", False):
            print("LoRA already applied, skip.")
            return
        print("Applying LoRA (safe) ...")

        for i, block in enumerate(self.blocks):
            base_path = f"blocks.{i}"
            self._apply_lora_recursive(block, base_path)

        self._lora_applied = True
        print(f"Applied LoRA to {len(self.lora_modules)} modules (unique).")

    def _freeze_first_lora_blocks(self, n: int):
        """硬冻结前 n 个 transformer block 的 LoRA 参数。
        对齐 eSCN stage2 `freeze_first_n_layers=K` 的语义：
          - 前 n 个 block：所有 LoRA A/B 参数 requires_grad=False（完全固定）
          - 剩余 block：LoRA 正常训练
        用于两阶段训练：stage2 保留 stage1 学好的前半段 LoRA 不变。
        """
        if n <= 0:
            return
        n = min(n, len(self.blocks))
        frozen_count = 0
        for i in range(n):
            block = self.blocks[i]
            for name, param in block.named_parameters():
                # 只冻结 LoRA 的 A/B 低秩矩阵（以及 enhanced TRF 的 C/D/T 等）
                if any(k in name for k in ['lora_A', 'lora_B', 'trf_']):
                    param.requires_grad = False
                    frozen_count += param.numel()
        logging.info(
            f"[EqV2-LoRA] freeze_first_n_lora_blocks={n}：冻结前 {n} 个 block 的 LoRA，"
            f"共 {frozen_count:,} 参数硬固定"
        )

    def _apply_lora_recursive(self, module: nn.Module, path: str):
        for child_name, child in list(module.named_children()):
            full_name = f"{path}.{child_name}" if path else child_name
            if child.__class__.__name__ == "LoRALinear":
                continue
            if isinstance(child, nn.Linear) and self._should_apply_lora_exact(full_name):
                self._replace_with_lora_safe(module, child_name, child, full_name)
                continue
            if isinstance(child, (nn.Sequential, nn.ModuleList)):
                for idx, sub_child in enumerate(child):
                    sub_name = f"{full_name}.{idx}"
                    if sub_child.__class__.__name__ == "LoRALinear":
                        continue
                    if isinstance(sub_child, nn.Linear) and self._should_apply_lora_exact(sub_name):
                        if self.use_enhanced_lora:
                            child[idx] = LoRATRFLinear(
                                original_layer=sub_child,
                                lora_rank=self.lora_rank,
                                lora_alpha=self.lora_alpha,
                                lora_dropout=self.lora_dropout,
                                trf_rank=self.trf_rank,
                                trf_alpha=self.trf_alpha,
                            )
                        else:
                            child[idx] = LoRALinear(
                                original_layer=sub_child,
                                rank=self.lora_rank,
                                alpha=self.lora_alpha,
                                dropout=self.lora_dropout
                            )
                        self.lora_modules[sub_name] = child[idx]
                        print(f"[LoRA] Wrapped: {sub_name}")
                    else:
                        self._apply_lora_recursive(sub_child, sub_name)
            else:
                self._apply_lora_recursive(child, full_name)

    def _should_apply_lora_exact(self, full_name: str) -> bool:
        parts = full_name.split(".")
        if parts[-1].isdigit():
            base = ".".join(parts[:-1])
        else:
            base = full_name
        for pattern in self.lora_config["target_modules"]:
            if base.endswith(pattern) or base == pattern:
                return True
        return False

    def _replace_with_lora_safe(self, parent: nn.Module, child_name: str, child_module: nn.Linear, full_name: str):
        if child_module.__class__.__name__ in ["LoRALinear", "LoRATRFLinear"]:
            return

        if self.use_enhanced_lora:
            new_layer = LoRATRFLinear(
                original_layer=child_module,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                trf_rank=self.trf_rank,
                trf_alpha=self.trf_alpha,
            )
        else:
            new_layer = LoRALinear(
                original_layer=child_module,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout
            )
        setattr(parent, child_name, new_layer)

        self.lora_modules[full_name] = new_layer
        print(f"[LoRA] Wrapped: {full_name}")

    def _freeze_base_model(self):
        frozen_params = 0
        trainable_params = 0
        lora_params = 0
        encoder_params = 0
        
        
        for name, param in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
                lora_params += param.numel()
                trainable_params += param.numel()
                continue
            
            should_train = any(module_name in name for module_name in self.trainable_modules)
            
            if should_train:
                param.requires_grad = True
                if 'mof_global_encoder' in name:
                    encoder_params += param.numel()
                trainable_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        total_params = frozen_params + trainable_params
        
        print("=" * 50)
        print("Parameter Freezing Summary:")
        print(f"Frozen parameters:           {frozen_params:,}")
        print(f"LoRA parameters:            {lora_params:,}")
        print(f"MOF encoder parameters:     {encoder_params:,}")
        print(f"Total trainable parameters: {trainable_params:,}")
        print(f"Total parameters:           {total_params:,}")
        print(f"Trainable ratio:            {trainable_params/total_params:.2%}")
        print("=" * 50)
    

    def load_pretrained_weights(self, checkpoint_path):
        
        map_location = torch.device("cpu") if self.cpu else self.device
        
        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.load_state_dict(checkpoint)
        self.eval()
        print('模型启动测试')

    def _compute_nm_per_node(self, data, num_atoms, device):
        """Compute nm (condition) encoding expanded to per-node.
        Returns [N, nm_encoded_dim] or None if disabled.
        """
        if not self.use_nm_features or self.condition_encoder is None:
            return None
        if not (hasattr(data, "condition") and data.condition is not None):
            return torch.zeros(num_atoms, self.nm_encoded_dim, device=device, dtype=data.pos.dtype)

        cond = data.condition
        if isinstance(cond, torch.Tensor):
            cond = cond.to(device=device, dtype=torch.float32)
            if cond.dim() == 1:
                cond = cond.view(1, -1)
        else:
            cond = torch.tensor(cond, device=device, dtype=torch.float32).view(1, -1)

        # Handle batched condition: [1, B*base_nm_dim] → [B, base_nm_dim]
        if cond.size(0) == 1 and cond.size(1) > self.base_nm_dim:
            n_cond = cond.size(1) // self.base_nm_dim
            if cond.size(1) % self.base_nm_dim == 0 and n_cond == len(data.natoms):
                cond = cond.view(n_cond, self.base_nm_dim)

        if cond.size(0) == 1 and len(data.natoms) > 1:
            cond = cond.expand(len(data.natoms), -1)

        nm_encoded = self.condition_encoder(cond)   # [B, nm_encoded_dim]
        batch_vec = data.batch if hasattr(data, "batch") else torch.zeros(num_atoms, dtype=torch.long, device=device)
        return nm_encoded[batch_vec]                # [N, nm_encoded_dim]

    def forward(self, data):
        mof_global_embedding = None
        if self.use_mof_global_features and self.mof_global_encoder is not None:
            mof_names_batch = []
            if hasattr(data, 'name') and data.name is not None:
                if isinstance(data.name, list):
                    for i in range(min(len(data.name), len(data.natoms))):
                        name = str(data.name[i])
                        mof_name = name.split('_')[0] if '_' in name else name
                        mof_names_batch.append(mof_name)
                    while len(mof_names_batch) < len(data.natoms):
                        mof_names_batch.append(mof_names_batch[-1] if len(mof_names_batch) > 0 else "")
                else:
                    name = str(data.name[0]) if hasattr(data.name, '__getitem__') else str(data.name)
                    mof_name = name.split('_')[0] if '_' in name else name
                    mof_names_batch = [mof_name] * len(data.natoms)
            if len(mof_names_batch) > 0:
                mof_embeddings_list = []
                for mof_name in mof_names_batch:
                    if mof_name:
                        emb = self.mof_global_encoder(mof_name)
                        mof_embeddings_list.append(emb)
                    else:
                        emb = torch.zeros(
                            1, 
                            self.mof_global_dim, 
                            device=data.pos.device, 
                            dtype=data.pos.dtype
                        )
                        mof_embeddings_list.append(emb)
                mof_global_embedding = torch.cat(mof_embeddings_list, dim=0)
        
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device
        atomic_numbers = data.atomic_numbers.long()
        
        graph = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        )
        data_batch = data.batch
        if gp_utils.initialized():
            (
                atomic_numbers,
                data_batch,
                node_offset,
                edge_index,
                edge_distance,
                edge_distance_vec,
            ) = self._init_gp_partitions(
                graph.atomic_numbers_full,
                graph.batch_full,
                graph.edge_index,
                graph.edge_distance,
                graph.edge_distance_vec,
            )
            graph.node_offset = node_offset
            graph.edge_index = edge_index
            graph.edge_distance = edge_distance
            graph.edge_distance_vec = edge_distance_vec
        
        edge_rot_mat = self._init_edge_rot_mat(
            data, graph.edge_index, graph.edge_distance_vec
        )
        
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)
        
        x = SO3_Embedding(
            len(atomic_numbers),
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )
        
        offset_res = 0
        offset = 0
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)
        
        graph.edge_distance = self.distance_expansion(graph.edge_distance)
        
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = graph.atomic_numbers_full[graph.edge_index[0]]
            target_element = graph.atomic_numbers_full[graph.edge_index[1]]
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            graph.edge_distance = torch.cat(
                (graph.edge_distance, source_embedding, target_embedding), dim=1
            )
        
        edge_degree = self.edge_degree_embedding(
            graph.atomic_numbers_full,
            graph.edge_distance,
            graph.edge_index,
            len(atomic_numbers),
            getattr(graph, 'node_offset', 0),
        )
        x.embedding = x.embedding + edge_degree.embedding
                                          
        if self.use_embedding_branch and self.embedding_branch_net is not None:
            with torch.no_grad() if self.freeze_embedding_branch else torch.enable_grad():
                branch_output = self.embedding_branch_net(data)
            branch_node_features = branch_output["node_features"]
            projected_branch_features = self.embedding_branch_proj(branch_node_features)
            x.embedding[:, 0, :] = x.embedding[:, 0, :] + projected_branch_features

        l0_features = x.embedding[:, 0, :].clone()  # [N, C]

        # nm(condition) per-node 编码，会参与下方各种 fusion 路径的 context
        nm_per_node = self._compute_nm_per_node(data, len(atomic_numbers), self.device)

        # atom feat helper（兼容 V1 / V2 的签名差异，V2 额外接受 tags）
        def _compute_atom_feat():
            if not (self.use_atom_extra_features and self.atom_encoder is not None):
                return None
            if self.atom_encoder_type == "v2":
                tags = data.tags if hasattr(data, "tags") else None
                return self.atom_encoder(atomic_numbers, tags=tags).to(device=self.device)
            else:
                return self.atom_encoder(atomic_numbers)

        # ---- 新增的注入/融合分支 ----
        if self.fusion_method == "concat":
            features_to_concat = [l0_features]
            if nm_per_node is not None:
                features_to_concat.append(nm_per_node)
            if self.use_mof_global_features and mof_global_embedding is not None:
                batch_vec = data.batch[:len(atomic_numbers)] if hasattr(data, 'batch') else torch.zeros(len(atomic_numbers), dtype=torch.long, device=self.device)
                global_per_node = mof_global_embedding[batch_vec]  # [N, mof_global_dim]
                features_to_concat.append(global_per_node)
            atom_feat = _compute_atom_feat()
            if atom_feat is not None:
                features_to_concat.append(atom_feat)
            fused_input = torch.cat(features_to_concat, dim=-1)
            if self.fusion_mlp is not None:
                enhanced_l0 = self.fusion_mlp(fused_input)      # 直接替换，不做 zero-init
            else:
                enhanced_l0 = l0_features

        elif self.fusion_method == "cross_attn":
            query = l0_features.unsqueeze(1)
            condition_features = []
            if self.use_mof_global_features and mof_global_embedding is not None and self.global_proj is not None:
                batch_vec = data.batch[:len(atomic_numbers)] if hasattr(data, 'batch') else torch.zeros(len(atomic_numbers), dtype=torch.long, device=self.device)
                global_per_node = mof_global_embedding[batch_vec]
                global_proj = self.global_proj(global_per_node)
                condition_features.append(global_proj.unsqueeze(1))
            atom_feat = _compute_atom_feat()
            if atom_feat is not None and self.atom_proj is not None:
                atom_proj = self.atom_proj(atom_feat)
                condition_features.append(atom_proj.unsqueeze(1))
            if condition_features:
                key_value = torch.cat(condition_features, dim=1)
                if self.cross_attn is None or self.fusion_proj is None:
                    enhanced_l0 = l0_features
                else:
                    enhanced_l0_raw, _attn = self.cross_attn(query=query, key=key_value, value=key_value)
                    enhanced_l0 = l0_features + self.fusion_proj(enhanced_l0_raw.squeeze(1))
            else:
                enhanced_l0 = l0_features

        elif self.fusion_method == "loratrf":
            context_features = []
            if nm_per_node is not None:
                context_features.append(nm_per_node)
            if self.use_mof_global_features and mof_global_embedding is not None:
                batch_vec = data.batch[:len(atomic_numbers)] if hasattr(data, 'batch') else torch.zeros(len(atomic_numbers), dtype=torch.long, device=self.device)
                global_per_node = mof_global_embedding[batch_vec]
                context_features.append(global_per_node)
            atom_feat = _compute_atom_feat()
            if atom_feat is not None:
                context_features.append(atom_feat)
            if context_features and self.loratrf_context_proj is not None:
                context_vec = torch.cat(context_features, dim=-1)
                task_vector = self.loratrf_context_proj(context_vec)
            elif self.loratrf_task_vector is not None:
                task_vector = self.loratrf_task_vector.expand(l0_features.size(0), -1)
            else:
                task_vector = None
            if task_vector is not None:
                similarity = F.cosine_similarity(l0_features, task_vector, dim=-1).unsqueeze(-1)
                similarity = torch.clamp(similarity, 0.0, 1.0)
                filtered_features = l0_features * similarity
                transformed_features = self.loratrf_transform_C(filtered_features)
                enhanced_update = self.loratrf_transform_D(transformed_features)
                enhanced_l0 = l0_features + enhanced_update
            else:
                enhanced_l0 = l0_features

        elif self.fusion_method in ["film", "gating", "add"] and self.context_input_dim > 0:
            # ── Context FiLM (nm + atom, eSCN/PaiNN 风格) ──
            # context 只包含 nm + atom，维度在 stage1/stage2 间保持一致
            # → film_gamma/beta、context_ln 能干净地从 stage1 checkpoint 加载
            context_features = []
            if nm_per_node is not None:
                context_features.append(nm_per_node)
            atom_feat = _compute_atom_feat()
            if atom_feat is not None:
                context_features.append(atom_feat)
            if context_features:
                context = torch.cat(context_features, dim=-1)
                if self.context_ln is not None:
                    context = self.context_ln(context)
                if self.fusion_method == "film":
                    gamma = self.film_gamma(context)
                    beta  = self.film_beta(context)
                    enhanced_l0 = l0_features * (1.0 + gamma) + beta
                elif self.fusion_method == "gating":
                    gate = self.gate_mlp(context)
                    enhanced_l0 = l0_features * (1.0 + gate)
                else:  # add
                    update = self.add_mlp(context)
                    enhanced_l0 = l0_features + update
            else:
                enhanced_l0 = l0_features
            # 注意：Global FiLM 不再在这里注入，改到 transformer loop 中层（参见下方 loop）
        else:
            enhanced_l0 = l0_features

        new_embedding = x.embedding.clone()
        new_embedding[:, 0, :] = enhanced_l0
        x.embedding = new_embedding

        # ── Global FiLM 所需的 per-node 全局编码（预计算一次，供循环中注入使用）──
        global_per_node_for_inject = None
        if (self.fusion_method == "film"
                and self.use_mof_global_features
                and mof_global_embedding is not None
                and self.global_gamma_mlp is not None):
            batch_vec = data.batch[:len(atomic_numbers)] if hasattr(data, 'batch') else torch.zeros(len(atomic_numbers), dtype=torch.long, device=self.device)
            global_per_node_for_inject = mof_global_embedding[batch_vec]  # [N, mof_global_dim]
            # GemNet 风格：注入前先对 global 做 LayerNorm（可选）
            if self.global_input_norm is not None:
                global_per_node_for_inject = self.global_input_norm(global_per_node_for_inject)

        # 保证与原版 EquiformerV2 一致的 batch/node_offset 传递
        if not hasattr(graph, 'node_offset'):
            graph.node_offset = 0
        for i in range(self.num_layers):
            # ── 中层 Global FiLM 注入：在 blocks[i] 之前（eSCN 风格 inject BEFORE layer K）──
            # global_inject_layer=0 → 在最开始（layer 0 前，与 context 同层注入）
            # global_inject_layer=4 → 经过 4 层 transformer 处理后再注入（推荐，8 层模型的中点）
            if (i == self.global_inject_layer
                    and global_per_node_for_inject is not None):
                gamma_g = self.global_gamma_mlp(global_per_node_for_inject)  # [N, C]
                beta_g  = self.global_beta_mlp(global_per_node_for_inject)
                # 只修改 l=0 分量（scalar），保持 l>0 等变张量不动
                l0_mid = x.embedding[:, 0, :]
                new_embedding = x.embedding.clone()
                new_embedding[:, 0, :] = l0_mid * (1.0 + gamma_g) + beta_g
                x.embedding = new_embedding

            x = self.blocks[i](
                x,
                graph.atomic_numbers_full,
                graph.edge_distance,
                graph.edge_index,
                batch=data_batch,
                node_offset=graph.node_offset,
            )

        # ── Post-loop Global FiLM 注入：当 global_inject_layer >= num_layers 时 ──
        # 语义：在所有 transformer blocks 之后、最终 norm 之前，对 l=0 分量施加 FiLM。
        # 用法：global_inject_layer=num_layers（本模型 =8）即 post-loop 注入。
        if (self.global_inject_layer >= self.num_layers
                and global_per_node_for_inject is not None):
            gamma_g = self.global_gamma_mlp(global_per_node_for_inject)
            beta_g  = self.global_beta_mlp(global_per_node_for_inject)
            l0_post = x.embedding[:, 0, :]
            new_embedding = x.embedding.clone()
            new_embedding[:, 0, :] = l0_post * (1.0 + gamma_g) + beta_g
            x.embedding = new_embedding

        x.embedding = self.norm(x.embedding)
        
        node_energy = self.energy_block(x)
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        
        energy = torch.zeros(
            len(data.natoms),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )
        energy.index_add_(0, getattr(graph, 'batch_full', data.batch), node_energy.view(-1))
        energy = energy / self.avg_num_nodes
        
        outputs = {"energy": energy}
        
        if self.regress_forces:
            forces = self.force_block(
                x,
                graph.atomic_numbers_full,
                graph.edge_distance,
                graph.edge_index,
                node_offset=getattr(graph, 'node_offset', 0),
            )
            forces = forces.embedding.narrow(1, 1, 3)
            forces = forces.view(-1, 3).contiguous()
            outputs["forces"] = forces
        
        return outputs
    
    def get_lora_parameters(self):
        """获取所有LoRA参数"""
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_params.append(param)
        return lora_params
    
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Linear, SO3_LinearV2)):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_LinearV2,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                    GaussianRadialBasisLayer,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, (torch.nn.Linear, SO3_LinearV2))
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)

# 省略其余辅助类/示例（与模型训练无关）
