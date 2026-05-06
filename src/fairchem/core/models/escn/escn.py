"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import logging
import time
import typing

import torch
import torch.nn as nn

if typing.TYPE_CHECKING:
    from torch_geometric.data.batch import Batch

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import BackboneInterface, GraphModelMixin, HeadInterface
from fairchem.core.models.escn.so3 import (
    CoefficientMapping,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
)
from fairchem.core.models.scn.sampling import CalcSpherePoints
from fairchem.core.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

with contextlib.suppress(ImportError):
    from e3nn import o3


class ConditionEncoder(nn.Module):
    """
    Encoder for condition features (e.g., molecule counts).
    Uses Gaussian Smearing to expand discrete counts into a continuous basis,
    followed by a small MLP.
    """
    def __init__(
        self,
        input_dim: int = 2,
        num_gaussians: int = 20,
        out_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        nm_max_count: float = 15.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_gaussians = num_gaussians
        self.out_dim = out_dim
        
        # Basis expansion for each input dimension
        self.smear = GaussianSmearing(start=0.0, stop=nm_max_count, num_gaussians=num_gaussians)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * num_gaussians, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        # condition: [B, input_dim]
        # Expand each dimension: [B, 1] -> [B, num_gaussians]
        smeared = []
        for i in range(self.input_dim):
            smeared.append(self.smear(condition[:, i]))
        
        # Concatenate smeared features: [B, input_dim * num_gaussians]
        x = torch.cat(smeared, dim=-1)
        return self.mlp(x)


@registry.register_model("escn")
class eSCN(nn.Module, GraphModelMixin):
    """Equivariant Spherical Channel Network
    Paper: Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs


    Args:
        use_pbc (bool):         Use periodic boundary conditions
        use_pbc_single (bool):         Process batch PBC graphs one at a time
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        cutoff (float):         Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        hidden_channels (int):        Number of hidden units in message passing
        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        edge_channels (int):          Number of channels for the edge invariant features
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        basis_width_scalar (float):   Width of distance basis function
        distance_resolution (float):  Distance between distance basis functions in Angstroms
        show_timing_info (bool):      Show timing and memory info
    """

    def __init__(
        self,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        regress_forces: bool = True,
        otf_graph: bool = False,
        max_neighbors: int = 40,
        cutoff: float = 8.0,
        max_num_elements: int = 90,
        num_layers: int = 8,
        lmax_list: list[int] | None = None,
        mmax_list: list[int] | None = None,
        sphere_channels: int = 128,
        hidden_channels: int = 256,
        edge_channels: int = 128,
        num_sphere_samples: int = 128,
        distance_function: str = "gaussian",
        basis_width_scalar: float = 1.0,
        distance_resolution: float = 0.02,
        show_timing_info: bool = False,
        resolution: int | None = None,
        activation_checkpoint: bool | None = False,
        # ---- 新增：原子级/全局特征注入（参考 equiformer_lora.py）----
        base_nm_dim: int = 2,  # 单个样本的 condition 特征维度
        use_atom_extra_features: bool = False,
        atom_extra_dim: int = 128,
        atom_encoder_type: str = "v1",  # "v1" = AtomPropertyEncoder, "v2" = AtomPropertyEncoderV2
        use_mof_global_features: bool = False,
        mof_global_excel_path: str | None = None,
        mof_global_dim: int = 128,
        # 选择编码器版本: "v1" = BiLSTM (原版), "v2" = 三模态预训练模型
        mof_global_encoder_type: str = "v1",
        # V2 专用路径（encoder_type="v2" 时生效）
        mof_smi_ted_folder: str | None = None,
        mof_selfies_ted_path: str | None = None,
        mof_mhg_path: str | None = None,
        mof_smiles_cache_path: str | None = None,
        adapter_hidden_dim: int = 128,
        adapter_dropout: float = 0.0,
        adapter_zero_init: bool = True,
        feature_fusion: str | None = "film",
        use_deep_injection: bool = False,
        # 全局特征中间层注入：0=第0层前（默认），N>0=第N层后
        global_inject_layer: int = 0,
        # 中间层注入模式："film"（默认）或 "attention"
        global_inject_mode: str = "film",
        # ---- 新增：NM 编码参数 ----
        nm_max_count: float = 15.0,
        condition_hidden_dim: int = 64,
        # 两阶段训练：冻结前 N 层 layer_blocks（0=不冻结）
        freeze_first_n_layers: int = 0,
    ) -> None:
        if mmax_list is None:
            mmax_list = [2]
        if lmax_list is None:
            lmax_list = [6]
        super().__init__()

        import sys

        if "e3nn" not in sys.modules:
            logging.error("You need to install the e3nn library to use the SCN model")
            raise ImportError

        self.activation_checkpoint = activation_checkpoint
        self.regress_forces = regress_forces

        # ---- 先初始化 eSCN 主干的基础属性（避免后面引用 self.xxx 不存在）----
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.show_timing_info = show_timing_info
        self.max_num_elements = max_num_elements
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_atoms = 0
        self.num_sphere_samples = num_sphere_samples
        self.sphere_channels = sphere_channels
        self.max_neighbors = max_neighbors
        self.edge_channels = edge_channels
        self.distance_resolution = distance_resolution
        self.grad_forces = False
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions: int = len(self.lmax_list)
        self.sphere_channels_all: int = self.num_resolutions * self.sphere_channels
        self.basis_width_scalar = basis_width_scalar
        self.distance_function = distance_function

        # ---- 新增：原子级/全局特征注入配置 ----
        self.use_atom_extra_features = bool(use_atom_extra_features)
        self.atom_extra_dim = int(atom_extra_dim)
        self.atom_encoder_type = atom_encoder_type.lower().strip()
        
        self.use_mof_global_features = bool(use_mof_global_features)
        self.mof_global_excel_path = mof_global_excel_path
        self.mof_global_dim = int(mof_global_dim)
        self.mof_global_encoder_type = mof_global_encoder_type.lower()
        self.mof_smi_ted_folder    = mof_smi_ted_folder
        self.mof_selfies_ted_path  = mof_selfies_ted_path
        self.mof_mhg_path          = mof_mhg_path
        self.mof_smiles_cache_path = mof_smiles_cache_path
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.adapter_dropout = float(adapter_dropout)
        self.adapter_zero_init = bool(adapter_zero_init)
        self.use_deep_injection = bool(use_deep_injection)

        # encoder/fusion modules
        self.atom_encoder = None
        self.mof_global_encoder = None
        self.context_ln = None
        self.film_gamma = None
        self.film_beta = None
        self.gate_mlp = None
        self.add_mlp = None

        # 原子属性 encoder（与 equiformer_lora.py 一致：输入 atomic_numbers -> [N, atom_extra_dim]）
        if self.use_atom_extra_features:
            try:
                if self.atom_encoder_type == "v2":
                    from fairchem.core.models.equiformer_v2.atomic_emb_v2 import AtomPropertyEncoderV2
                    self.atom_encoder = AtomPropertyEncoderV2(
                        max_Z=self.max_num_elements, out_dim=self.atom_extra_dim
                    )
                else:
                    from fairchem.core.models.equiformer_v2.atomic_emb import AtomPropertyEncoder
                    self.atom_encoder = AtomPropertyEncoder(
                        max_Z=self.max_num_elements, out_dim=self.atom_extra_dim
                    )
            except Exception as e:
                logging.warning(f"Failed to init AtomPropertyEncoder (type={self.atom_encoder_type}): {e}")
                self.atom_encoder = None

        # MOF 全局 encoder（输入 mof_name -> [1, mof_global_dim]）
        if self.use_mof_global_features and self.mof_global_excel_path:
            _dev = "cuda" if torch.cuda.is_available() else "cpu"
            if self.mof_global_encoder_type == "v2":
                try:
                    from fairchem.core.models.equiformer_v2.global_emb_v2 import MOFGlobalEncoderV2
                    self.mof_global_encoder = MOFGlobalEncoderV2(
                        excel_path        = self.mof_global_excel_path,
                        smi_ted_folder    = self.mof_smi_ted_folder,
                        selfies_ted_path  = self.mof_selfies_ted_path,
                        mhg_path          = self.mof_mhg_path,
                        global_dim        = self.mof_global_dim,
                        device            = _dev,
                        cache_path        = self.mof_smiles_cache_path,
                    )
                    logging.info("MOFGlobalEncoderV2 (三模态) 初始化成功")
                except Exception as e:
                    logging.warning(f"Failed to init MOFGlobalEncoderV2: {e}. Falling back to v1.")
                    self.mof_global_encoder_type = "v1"

            if self.mof_global_encoder_type == "v1":
                try:
                    from fairchem.core.models.equiformer_v2.global_emb import MOFGlobalPropertyEncoder
                    self.mof_global_encoder = MOFGlobalPropertyEncoder(
                        excel_path = self.mof_global_excel_path,
                        global_dim = self.mof_global_dim,
                        hidden_dim = max(256, self.mof_global_dim * 2),
                        device     = _dev,
                    )
                    logging.info("MOFGlobalPropertyEncoder (BiLSTM v1) 初始化成功")
                except Exception as e:
                    logging.warning(f"Failed to init MOFGlobalPropertyEncoder: {e}")
                    self.mof_global_encoder = None

        # ---- 简化版：拼接 [l0(128) || context(50)] -> MLP(178->128) 残差注入 ----
        # nm(condition) 固定 2 维；全局 32 维；原子 16 维
        self.base_nm_dim = base_nm_dim
        self.use_nm_features = True
        self.global_inject_layer = int(global_inject_layer)
        self.global_inject_mode = global_inject_mode.lower() if global_inject_mode else "film"
        self.freeze_first_n_layers = int(freeze_first_n_layers)

        # 新增：融合/注入模式（默认 film）
        self.feature_fusion = feature_fusion.lower() if feature_fusion else "film"
        if self.feature_fusion not in ["film", "concat", "gating", "add", "phast"]:
             logging.warning(f"Unknown feature_fusion '{self.feature_fusion}', defaulting to 'film'")
             self.feature_fusion = "film"

        # 兼容旧字段名，避免其它代码引用报错
        self.context_ln = None
        self.context_mlp = None # Deprecated, mapped to new modules below

        # Variables used for display purposes
        self.counter = 0

        # Non-linear activation function used throughout the network
        self.act = nn.SiLU()
        
        # Register backward compatibility hook for loading old checkpoints
        self._register_load_state_dict_pre_hook(self._backward_compat_hook)


        # Initialize Condition Encoder (NM)
        self.condition_encoder = None
        self.nm_max_count = float(nm_max_count)
        self.tgt_condition_hidden_dim = int(condition_hidden_dim)

        if self.use_nm_features:
             self.condition_encoder = ConditionEncoder(
                 input_dim=self.base_nm_dim,
                 num_gaussians=20, # Default expansion
                 out_dim=self.tgt_condition_hidden_dim, 
                 hidden_dim=self.adapter_hidden_dim,
                 dropout=self.adapter_dropout,
                 nm_max_count=self.nm_max_count
             )
             self.nm_encoded_dim = self.condition_encoder.out_dim
             # Re-calculate context_input_dim with encoded dimension
             # context = [encoded_nm || global(仅当global_inject_layer==0) || atom]
             self.context_input_dim = self.nm_encoded_dim
             if self.use_mof_global_features and self.mof_global_encoder is not None:
                  if self.global_inject_layer == 0:
                      self.context_input_dim += self.mof_global_encoder.global_dim
             if self.use_atom_extra_features and self.atom_encoder is not None:
                  self.context_input_dim += self.atom_extra_dim
        
        # Initialize Context LayerNorm
        if self.context_input_dim > 0:
             if self.feature_fusion == "concat":
                 # Heuristic: If context dim is small (Legacy/Baseline, e.g. 2), it follows old logic (LN on Fused).
                 # If context dim is large (Global, e.g. 194), it follows new logic (LN on Context only).
                 if self.context_input_dim < 10:
                     self.ln_on_fused = True 
                     self.context_ln = nn.LayerNorm(self.sphere_channels + self.context_input_dim)
                 else:
                     self.ln_on_fused = False
                     self.context_ln = nn.LayerNorm(self.context_input_dim)
             else:
                 # Standard non-concat modes (like FiLM) always LN on context
                 self.ln_on_fused = False
                 self.context_ln = nn.LayerNorm(self.context_input_dim)
        
        # Initialize Fusion Modules
        self.film_gamma = None
        self.film_beta = None
        self.gate_mlp = None
        self.add_mlp = None
        self.fused_mlp = None # Renamed/Refactored from original "fused_mlp" for "concat" mode
        
        # Deep Injection layers
        self.film_gamma_layers = None
        self.film_beta_layers = None
        self.film_norm_layers = None

        def _zero_last_linear(module):
            if isinstance(module, nn.Sequential):
                for m in reversed(module):
                    if isinstance(m, nn.Linear):
                        nn.init.zeros_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                        break
        
        if self.context_input_dim > 0:
            if self.feature_fusion == "film":
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

                if self.use_deep_injection:
                    self.film_gamma_layers = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(self.context_input_dim, self.adapter_hidden_dim),
                            nn.LayerNorm(self.adapter_hidden_dim),
                            nn.GELU(),
                            nn.Dropout(self.adapter_dropout),
                            nn.Linear(self.adapter_hidden_dim, self.sphere_channels),
                        ) for _ in range(self.num_layers)
                    ])
                    self.film_beta_layers = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(self.context_input_dim, self.adapter_hidden_dim),
                            nn.LayerNorm(self.adapter_hidden_dim),
                            nn.GELU(),
                            nn.Dropout(self.adapter_dropout),
                            nn.Linear(self.adapter_hidden_dim, self.sphere_channels),
                        ) for _ in range(self.num_layers)
                    ])
                    # AdaLN Norm layers for stability
                    self.film_norm_layers = nn.ModuleList([
                        nn.LayerNorm(self.sphere_channels) for _ in range(self.num_layers)
                    ])

                    if self.adapter_zero_init:
                        for m in self.film_gamma_layers:
                            _zero_last_linear(m)
                        for m in self.film_beta_layers:
                            _zero_last_linear(m)
            
            elif self.feature_fusion == "gating":
                 self.gate_mlp = nn.Sequential(
                    nn.Linear(self.context_input_dim, self.adapter_hidden_dim),
                    nn.LayerNorm(self.adapter_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.adapter_dropout),
                    nn.Linear(self.adapter_hidden_dim, self.sphere_channels),
                    nn.Tanh(),
                )
                 if self.adapter_zero_init:
                     # Zero init the Linear before Tanh
                     for m in reversed(self.gate_mlp):
                        if isinstance(m, nn.Linear):
                            nn.init.zeros_(m.weight)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)
                            break

            elif self.feature_fusion == "add":
                self.add_mlp = nn.Sequential(
                    nn.Linear(self.context_input_dim, self.adapter_hidden_dim),
                    nn.LayerNorm(self.adapter_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.adapter_dropout),
                    nn.Linear(self.adapter_hidden_dim, self.sphere_channels),
                )
                if self.adapter_zero_init:
                    _zero_last_linear(self.add_mlp)

            elif self.feature_fusion == "concat":
                # Original "concat" implementation logic
                # [N, C + context] -> [N, C]
                self.fused_dim = self.sphere_channels + self.context_input_dim
                self.fused_mlp = nn.Sequential(
                    nn.Linear(self.fused_dim, self.adapter_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.adapter_dropout),
                    nn.Linear(self.adapter_hidden_dim, self.sphere_channels),
                )
                if self.adapter_zero_init:
                     _zero_last_linear(self.fused_mlp)

            elif self.feature_fusion == "phast":
                # PhAST: [N, context] -> [N, C] (Direct replacement, no old L0)
                self.fused_dim = self.context_input_dim
                self.fused_mlp = nn.Sequential(
                    nn.Linear(self.fused_dim, self.adapter_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.adapter_dropout),
                    nn.Linear(self.adapter_hidden_dim, self.sphere_channels),
                )

        # ── 中间层全局特征注入模块（global_inject_layer > 0 时生效）────────────
        # mode="film":      l0' = l0*(1+γ) + β，γ/β 由 global 生成，zero-init→恒等
        # mode="attention": l0' = l0 + W_out(a_i * V)，Q=linear(l0)，K/V=linear(global)
        #                   W_out zero-init → 恒等，训练后每个原子以不同权重关注全局
        self.global_gamma_mlp   = None   # FiLM gamma
        self.global_beta_mlp    = None   # FiLM beta
        self.global_attn_q      = None   # Attention Q projection
        self.global_attn_k      = None   # Attention K projection
        self.global_attn_v      = None   # Attention V projection
        self.global_attn_out    = None   # Attention output projection (zero-init)
        self.global_attn_norm   = None   # pre-LN for attention input

        if (self.global_inject_layer > 0 and
                self.use_mof_global_features and
                self.mof_global_encoder is not None):
            _g_dim = self.mof_global_encoder.global_dim
            _C     = self.sphere_channels

            if self.global_inject_mode == "attention":
                # Cross-attention：原子 l0 作为 Q，全局向量作为 K/V
                # 单头，head_dim = adapter_hidden_dim
                _d = self.adapter_hidden_dim
                self.global_attn_norm = nn.LayerNorm(_C)
                self.global_attn_q    = nn.Linear(_C,    _d, bias=False)
                self.global_attn_k    = nn.Linear(_g_dim, _d, bias=False)
                self.global_attn_v    = nn.Linear(_g_dim, _d, bias=False)
                self.global_attn_out  = nn.Linear(_d,    _C, bias=True)
                self._attn_scale      = _d ** -0.5
                if self.adapter_zero_init:
                    nn.init.zeros_(self.global_attn_out.weight)
                    nn.init.zeros_(self.global_attn_out.bias)
                logging.info(f"[global_inject] layer={self.global_inject_layer}, mode=attention, d={_d}")
            else:  # "film" (default)
                self.global_gamma_mlp = nn.Sequential(
                    nn.Linear(_g_dim, self.adapter_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.adapter_hidden_dim, _C),
                )
                self.global_beta_mlp = nn.Sequential(
                    nn.Linear(_g_dim, self.adapter_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.adapter_hidden_dim, _C),
                )
                if self.adapter_zero_init:
                    _zero_last_linear(self.global_gamma_mlp)
                    _zero_last_linear(self.global_beta_mlp)
                logging.info(f"[global_inject] layer={self.global_inject_layer}, mode=film")

        # variables used for display purposes
        self.counter = 0

        # non-linear activation function used throughout the network
        self.act = nn.SiLU()

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels_all
        )

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            "gaussian",
            "sigmoid",
            "linearsigmoid",
            "silu",
        ]
        self.num_gaussians = int(cutoff / self.distance_resolution)
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "sigmoid":
            self.distance_expansion = SigmoidSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "linearsigmoid":
            self.distance_expansion = LinearSigmoidSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "silu":
            self.distance_expansion = SiLUSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = nn.ModuleList()
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(SO3_Grid(lval, m, resolution=resolution))

            self.SO3_grid.append(SO3_m_grid)

        # Initialize the blocks for each layer of the GNN
        self.layer_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = LayerBlock(
                i,
                self.sphere_channels,
                self.hidden_channels,
                self.edge_channels,
                self.lmax_list,
                self.mmax_list,
                self.distance_expansion,
                self.max_num_elements,
                self.SO3_grid,
                self.act,
            )
            self.layer_blocks.append(block)

        # Output blocks for energy and forces
        self.energy_block = EnergyBlock(
            self.sphere_channels_all, self.num_sphere_samples, self.act
        )
        if self.regress_forces:
            self.force_block = ForceBlock(
                self.sphere_channels_all, self.num_sphere_samples, self.act
            )

        # Create a roughly evenly distributed point sampling of the sphere for the output blocks
        self.sphere_points = nn.Parameter(
            CalcSpherePoints(self.num_sphere_samples), requires_grad=False
        )

        # For each spherical point, compute the spherical harmonic coefficient weights
        sphharm_weights: list[nn.Parameter] = []
        for i in range(self.num_resolutions):
            sphharm_weights.append(
                nn.Parameter(
                    o3.spherical_harmonics(
                        torch.arange(0, self.lmax_list[i] + 1).tolist(),
                        self.sphere_points,
                        False,
                    ),
                    requires_grad=False,
                )
            )
        self.sphharm_weights = nn.ParameterList(sphharm_weights)

        # 两阶段训练：冻结前 N 层及其输入模块
        if self.freeze_first_n_layers > 0:
            n = self.freeze_first_n_layers
            # 冻结输入嵌入和距离编码（这些模块为前 N 层提供输入，一并冻结保证 l0 稳定）
            self.sphere_embedding.requires_grad_(False)
            self.distance_expansion.requires_grad_(False)
            # 冻结前 N 个 layer_blocks
            for i in range(min(n, self.num_layers)):
                self.layer_blocks[i].requires_grad_(False)
            logging.info(f"[freeze] Froze sphere_embedding, distance_expansion, and layer_blocks[0:{n}]")

    def _backward_compat_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Auto-rename legacy keys (fused_ln) to new keys (context_ln) to support loading old checkpoints.
        """
        keys = list(state_dict.keys())
        for k in keys:
            if "fused_ln" in k:
                new_k = k.replace("fused_ln", "context_ln")
                # Only remap if the target key expects a replacement (and fits new key availability)
                if new_k not in state_dict:
                     state_dict[new_k] = state_dict.pop(k)
                     # logging.info(f"Remapped legacy key: {k} -> {new_k}")

    def forward_embeddings(self, data):
        """只计算 eSCN 的中间表示（不做 energy/forces 头）。

        返回 dict，至少包含：
          - sphere_values: (num_atoms, sphere_channels_all)
          - sphere_points: (num_sphere_samples, 3)
        """
        device = data.pos.device
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype

        start_time = time.time()
        atomic_numbers = data.atomic_numbers.long()
        assert (
            atomic_numbers.max().item() < self.max_num_elements
        ), "Atomic number exceeds that given in model config"
        num_atoms = len(atomic_numbers)
        graph = self.generate_graph(data)

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, graph.edge_index, graph.edge_distance_vec
        )

        # Initialize the WignerD matrices
        SO3_edge_rot = []
        for i in range(self.num_resolutions):
            SO3_edge_rot.append(SO3_Rotation(edge_rot_mat, self.lmax_list[i]).to(device=device, dtype=self.dtype))

        # Init per node representations
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        for i in range(self.num_resolutions):
            x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                :, offset : offset + self.sphere_channels
            ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # ---- 简化版：拼接 [l0_old || context] -> MLP -> 残差注入到 l=0 ----
        global_per_node = None  # 供中间层注入使用
        if self.context_input_dim > 0:
            context_features = []

            # 0) nm(condition) -> per-node
            if hasattr(data, "condition") and data.condition is not None:
                cond = data.condition
                if isinstance(cond, torch.Tensor):
                    cond = cond.to(device=device, dtype=torch.float32)
                    if cond.dim() == 1:
                        # [2] -> [1,2]
                        cond = cond.view(1, -1)
                else:
                    cond = torch.tensor(cond, device=device, dtype=torch.float32).view(1, -1)

                # 自动处理因 batch collate 导致的 condition 维度变化
                # cond 初始可能是 [1, B * base_nm_dim], B=batch_size
                if cond.size(0) == 1 and cond.size(1) > self.base_nm_dim:
                    num_conditions = cond.size(1) // self.base_nm_dim
                    if cond.size(1) % self.base_nm_dim == 0 and num_conditions == len(data.natoms):
                        # [1, B * base_nm_dim] -> [B, base_nm_dim]
                        cond = cond.view(num_conditions, self.base_nm_dim)

                if cond.size(-1) != self.base_nm_dim:
                    raise ValueError(f"data.condition last dim must be {self.base_nm_dim}, got {tuple(cond.shape)}")

                if cond.size(0) == 1 and len(data.natoms) > 1:
                    # broadcast to batch size
                    cond = cond.expand(len(data.natoms), -1)

                batch_indices = data.batch if hasattr(data, "batch") else torch.zeros(num_atoms, dtype=torch.long, device=device)
                nm_encoded = self.condition_encoder(cond) # [B, encoded_dim]
                nm_per_node = nm_encoded[batch_indices]  # [N, encoded_dim]
                context_features.append(nm_per_node)
            else:
                # 没有 condition 就用 0 占位，保证维度一致
                context_features.append(torch.zeros(num_atoms, self.nm_encoded_dim, device=device, dtype=torch.float32))

            # 1) 全局特征（通过 data.name 查表） -> per-node
            if self.use_mof_global_features and self.mof_global_encoder is not None:
                mof_embeddings = []
                if hasattr(data, "name") and data.name is not None:
                    if isinstance(data.name, (list, tuple)):
                        for name in data.name:
                            name = str(name)
                            mof_name = name.split("_")[0] if "_" in name else name
                            mof_embeddings.append(self.mof_global_encoder(mof_name))
                        while len(mof_embeddings) < len(data.natoms):
                            mof_embeddings.append(mof_embeddings[-1] if len(mof_embeddings) > 0 else torch.zeros(1, self.mof_global_dim, device=device))
                    else:
                        name = str(data.name[0]) if isinstance(data.name, (list, tuple)) else str(data.name)
                        mof_name = name.split("_")[0] if "_" in name else name
                        mof_embeddings = [self.mof_global_encoder(mof_name)] * len(data.natoms)

                if mof_embeddings:
                    global_emb = torch.cat(mof_embeddings, dim=0).to(device=device)  # [B,G]
                    global_emb = torch.nan_to_num(global_emb, nan=0.0, posinf=0.0, neginf=0.0)
                    batch_indices = data.batch if hasattr(data, "batch") else torch.zeros(num_atoms, dtype=torch.long, device=device)
                    global_per_node = global_emb[batch_indices]
                    if self.global_inject_layer == 0:
                        # 正常：加入早期 context 一起注入
                        context_features.append(global_per_node)
                    # else: global_per_node 留给中间层注入，不加入 context_features

            # 2) 原子级特征 (PhAST style: includes Tag embedding)
            if self.use_atom_extra_features and self.atom_encoder is not None:
                tags = data.tags if hasattr(data, "tags") else None
                atom_feat = self.atom_encoder(atomic_numbers, tags=tags).to(device=device)  # [N,A]
                context_features.append(atom_feat)

            # 1. 原始 l0 特征
            l0_features = x.embedding[:, 0, :].clone()  # [N, C]

            # Debug print
            if self.counter % 1000 == 0:
                 print(f"[Debug] Step {self.counter}: l0_features min={l0_features.min():.4f}, max={l0_features.max():.4f}, mean={l0_features.mean():.4f}")

            # 2. 上下文特征
            context = torch.cat(context_features, dim=-1)  # [N, D_context]
            
            # For non-concat modes, apply LN to context here
            if self.context_ln is not None and self.feature_fusion != "concat":
                context = self.context_ln(context)

            enhanced_l0 = l0_features

            if self.counter % 1000 == 0:
                 print(f"[Debug] Step {self.counter}: Context min={context.min():.4f}, max={context.max():.4f}, mean={context.mean():.4f} shape={context.shape}")

            if self.feature_fusion == "film" and self.film_gamma is not None:
                gamma = self.film_gamma(context)
                beta = self.film_beta(context)
                enhanced_l0 = l0_features * (1.0 + gamma) + beta
            
            elif self.feature_fusion == "gating" and self.gate_mlp is not None:
                gate = self.gate_mlp(context)
                enhanced_l0 = l0_features * (1.0 + gate)
            
            elif self.feature_fusion == "add" and self.add_mlp is not None:
                update = self.add_mlp(context)
                enhanced_l0 = l0_features + update

            elif self.feature_fusion == "concat" and self.fused_mlp is not None:
                 # concat logic: cat([l0_norm, context_norm]) -> mlp -> delta -> l0 + delta

                 # Logic A: New Global Checkpoint (LN on Context Only)
                 if self.context_ln is not None and not self.ln_on_fused:
                     context = self.context_ln(context)

                 fused_input = torch.cat([l0_features, context], dim=-1)

                 # Logic B: Old Baseline Checkpoint (LN on Fused Vector)
                 if self.context_ln is not None and self.ln_on_fused:
                     fused_input = self.context_ln(fused_input)
                     
                 update = self.fused_mlp(fused_input)
                 enhanced_l0 = l0_features + update
            
            elif self.feature_fusion == "phast" and self.fused_mlp is not None:
                 # PhAST Path: Direct Projection (No Residual, No legacy L0)
                 # [N, D] (Context) -> MLP -> [N, C]
                 fused_input = context
                 enhanced_l0 = self.fused_mlp(fused_input)

            x.embedding = x.embedding.clone() 
            x.embedding[:, 0, :] = enhanced_l0
            
            self.counter += 1

        mappingReduced = CoefficientMapping(self.lmax_list, self.mmax_list, device)

        for i in range(self.num_layers):
            # ── 中间层全局特征注入 ──────────────────────────────────────────────
            if (i == self.global_inject_layer and
                    self.global_inject_layer > 0 and
                    global_per_node is not None):
                l0_mid = x.embedding[:, 0, :].clone()  # [N, C]
                x.embedding = x.embedding.clone()
                if self.global_inject_mode == "attention" and self.global_attn_q is not None:
                    # Cross-attention：Q=f(l0)，K/V=f(global)
                    # pre-LN 稳定 Q 的尺度
                    q = self.global_attn_q(self.global_attn_norm(l0_mid))  # [N, d]
                    k = self.global_attn_k(global_per_node)                # [N, d]
                    v = self.global_attn_v(global_per_node)                # [N, d]
                    # 点积注意力权重（每个原子对全局向量的标量权重）
                    attn_w = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * self._attn_scale)  # [N,1]
                    update = self.global_attn_out(attn_w * v)              # [N, C]，初始≈0
                    x.embedding[:, 0, :] = l0_mid + update
                elif self.global_gamma_mlp is not None:
                    # FiLM
                    gamma = self.global_gamma_mlp(global_per_node)  # [N, C]，初始≈0
                    beta  = self.global_beta_mlp(global_per_node)   # [N, C]，初始≈0
                    x.embedding[:, 0, :] = l0_mid * (1.0 + gamma) + beta
            # ────────────────────────────────────────────────────────────────────

            if self.activation_checkpoint:
                x_message = torch.utils.checkpoint.checkpoint(
                    self.layer_blocks[i],
                    x,
                    atomic_numbers,
                    graph.edge_distance,
                    graph.edge_index,
                    SO3_edge_rot,
                    mappingReduced,
                    use_reentrant=not self.training,
                )
            else:
                x_message = self.layer_blocks[i](
                    x,
                    atomic_numbers,
                    graph.edge_distance,
                    graph.edge_index,
                    SO3_edge_rot,
                    mappingReduced,
                )

            # --- Deep Injection (Layer-wise FiLM) ---
            # Apply to the UPDATE (x_message) rather than the ACCUMULATOR (x) for stability
            # [Stabilization] Use AdaLN: Norm(x) -> Scale/Shift
            if self.use_deep_injection and self.feature_fusion == "film" and self.context_input_dim > 0:
                if self.film_gamma_layers is not None and self.film_beta_layers is not None and self.film_norm_layers is not None:
                     l0_msg = x_message.embedding[:, 0, :]
                     
                     # 1. Apply LayerNorm to stabilize the incoming update signal
                     l0_msg_norm = self.film_norm_layers[i](l0_msg)

                     # 2. Compute parameters (with Tanh gating for bounded stability)
                     gamma_deep = self.film_gamma_layers[i](context).tanh()
                     beta_deep = self.film_beta_layers[i](context)
                     
                     # 3. Modulate: (Normed Input) * (1 + gamma) + beta
                     # Clone to avoid inplace error
                     x_message.embedding = x_message.embedding.clone()
                     x_message.embedding[:, 0, :] = l0_msg_norm * (1.0 + gamma_deep) + beta_deep

            if i > 0:
                x.embedding = x.embedding + x_message.embedding
            else:
                x = x_message

        # Sample spherical channels at sphere points
        x_pt_list = []
        offset = 0
        for i in range(self.num_resolutions):
            num_coefficients = int((x.lmax_list[i] + 1) ** 2)
            x_pt_list.append(
                torch.einsum(
                    "abc, pb->apc",
                    x.embedding[:, offset : offset + num_coefficients],
                    self.sphharm_weights[i],
                ).contiguous()
            )
            offset = offset + num_coefficients

        x_pt = torch.cat(x_pt_list, dim=2).view(-1, self.sphere_channels_all)

        if self.show_timing_info is True:
            torch.cuda.synchronize()
            logging.info(
                f"{self.counter} Time: {time.time() - start_time}\tMemory: {len(data.pos)}\t{torch.cuda.max_memory_allocated() / 1000000}"
            )

        return {
            "sphere_values": x_pt,
            "sphere_points": self.sphere_points,
            "graph": graph,
        }

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        device = data.pos.device
        # 直接复用 forward_embeddings，避免重复计算（不影响结果）
        emb = self.forward_embeddings(data)
        x_pt = emb["sphere_values"]

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x_pt)
        energy = torch.zeros(len(data.natoms), device=device)
        energy.index_add_(0, data.batch, node_energy.view(-1))
        # Scale energy to help balance numerical precision w.r.t. forces
        # energy = energy * 0.001

        outputs = {"energy": energy}
        ###############################################################
        # Force estimation
        ###############################################################
        if self.regress_forces:
            forces = self.force_block(x_pt, self.sphere_points)
            outputs["forces"] = forces

        return outputs

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        edge_vec_0 = edge_distance_vec
        edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

        # Make sure the atoms are far enough apart
        if torch.min(edge_vec_0_distance) < 0.0001:
            logging.error(
                f"Error edge_vec_0_distance: {torch.min(edge_vec_0_distance)}"
            )
            (minval, minidx) = torch.min(edge_vec_0_distance, 0)
            logging.error(
                f"Error edge_vec_0_distance: {minidx} {edge_index[0, minidx]} {edge_index[1, minidx]} {data.pos[edge_index[0, minidx]]} {data.pos[edge_index[1, minidx]]}"
            )

        norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

        edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
        edge_vec_2 = edge_vec_2 / (
            torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
        )
        # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
        # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
        edge_vec_2b = edge_vec_2.clone()
        edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
        edge_vec_2b[:, 1] = edge_vec_2[:, 0]
        edge_vec_2c = edge_vec_2.clone()
        edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
        edge_vec_2c[:, 2] = edge_vec_2[:, 1]
        vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
        vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
        # Check the vectors aren't aligned
        assert torch.max(vec_dot) < 0.99

        norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
        norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True)))
        norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1))
        norm_y = torch.cross(norm_x, norm_z, dim=1)
        norm_y = norm_y / (torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True)))

        # Construct the 3D rotation matrix
        norm_x = norm_x.view(-1, 3, 1)
        norm_y = -norm_y.view(-1, 3, 1)
        norm_z = norm_z.view(-1, 3, 1)

        edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
        edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

        return edge_rot_mat.detach()

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@registry.register_model("escn_backbone")
class eSCNBackbone(eSCN, BackboneInterface):
    @conditional_grad(torch.enable_grad())
    def forward(self, data: Batch) -> dict[str, torch.Tensor]:
        device = data.pos.device
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype

        atomic_numbers = data.atomic_numbers.long()
        num_atoms = len(atomic_numbers)

        graph = self.generate_graph(data)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, graph.edge_index, graph.edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        SO3_edge_rot = []
        for i in range(self.num_resolutions):
            SO3_edge_rot.append(SO3_Rotation(edge_rot_mat, self.lmax_list[i]).to(device=device, dtype=self.dtype))

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l=0,m=0 coefficients for each resolution
        for i in range(self.num_resolutions):
            x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                :, offset : offset + self.sphere_channels
            ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # This can be expensive to compute (not implemented efficiently), so only do it once and pass it along to each layer
        mappingReduced = CoefficientMapping(self.lmax_list, self.mmax_list, device)

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            if i > 0:
                x_message = self.layer_blocks[i](
                    x,
                    atomic_numbers,
                    graph.edge_distance,
                    graph.edge_index,
                    SO3_edge_rot,
                    mappingReduced,
                )

                # Residual layer for all layers past the first
                x.embedding = x.embedding + x_message.embedding

            else:
                # No residual for the first layer
                x = self.layer_blocks[i](
                    x,
                    atomic_numbers,
                    graph.edge_distance,
                    graph.edge_index,
                    SO3_edge_rot,
                    mappingReduced,
                )

        # Sample the spherical channels (node embeddings) at evenly distributed points on the sphere.
        # These values are fed into the output blocks.
        # Compute the embedding values at every sampled point on the sphere
        x_pt_list = []
        offset = 0
        for i in range(self.num_resolutions):
            num_coefficients = int((x.lmax_list[i] + 1) ** 2)
            x_pt_list.append(
                torch.einsum(
                    "abc, pb->apc",
                    x.embedding[:, offset : offset + num_coefficients],
                    self.sphharm_weights[i],
                ).contiguous()
            )
            offset = offset + num_coefficients

        x_pt = torch.cat(x_pt_list, dim=2).view(-1, self.sphere_channels_all)

        return {
            "sphere_values": x_pt,
            "sphere_points": self.sphere_points,
            "node_embedding": x,
            "graph": graph,
        }


@registry.register_model("escn_energy_head")
class eSCNEnergyHead(nn.Module, HeadInterface):
    def __init__(
        self,
        backbone,
        reduce="sum",
        condition_dim: int = 2,
        condition_hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__()

        # 兼容两种传参方式：
        # 1) backbone 是已实例化的 eSCN 模型对象（老用法）
        # 2) backbone 是 dict 配置（yml 里常见写法），这里自动实例化
        if isinstance(backbone, dict):
            bb_cfg = dict(backbone)
            bb_name = bb_cfg.pop("name", "escn")
            backbone = registry.get_model_class(bb_name)(**bb_cfg)

        # 保存 backbone，使得本 head 可以作为完整模型被 trainer 直接调用：model(batch)
        self.backbone = backbone

        # 注意：不要把 backbone.energy_block 置空；我们调用 backbone.forward_embeddings 来绕开 backbone 自带能量头
        self.reduce = reduce
        self.condition_dim = condition_dim
        # Output blocks for energy and forces
        self.energy_block = EnergyBlock(
            backbone.sphere_channels_all, backbone.num_sphere_samples, backbone.act
        )

    def forward(
        self, data: Batch, emb: dict[str, torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        # 如果外部没有传入 emb，则自己调用 backbone 生成
        if emb is None:
            # backbone(escn) 默认 forward 返回的是 {"energy": ...}
            # 我们需要的是中间表征，因此调用 forward_embeddings
            if hasattr(self.backbone, "forward_embeddings"):
                emb = self.backbone.forward_embeddings(data)
            else:
                raise AttributeError(
                    "backbone does not have forward_embeddings; cannot build sphere_values for head"
                )

        # 计算节点能量并聚合
        node_energy = self.energy_block(emb["sphere_values"])
        energy = torch.zeros(len(data.natoms), device=data.pos.device)
        energy.index_add_(0, data.batch, node_energy.view(-1))


        if self.reduce == "sum":
            # Scale energy to help balance numerical precision w.r.t. forces
            return {"energy": energy}
        elif self.reduce == "mean":
            return {"energy": energy / data.natoms}
        else:
            raise ValueError(
                f"reduce can only be sum or mean, user provided: {self.reduce}"
            )




@registry.register_model("escn_weighted_energy_head")
class eSCNWeightedEnergyHead(nn.Module, HeadInterface):
    """
    PhAST-style weighted energy head.
    
    Instead of simple summation, learns per-atom weights:
    E_total = sum_i (w_i * e_i)
    
    This allows the model to learn that certain atoms (e.g., active sites,
    adsorbates) contribute more to the total energy than bulk atoms.
    
    Reference: PhAST paper (Duval et al., 2024), Section 3.3
    """
    def __init__(
        self,
        backbone,
        reduce="weighted_sum",
        weight_nn_hidden_dim=64,
        use_initial_embeddings=True,
        condition_dim: int = 2,
        condition_hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        
        # Handle backbone initialization
        if isinstance(backbone, dict):
            bb_cfg = dict(backbone)
            bb_name = bb_cfg.pop("name", "escn")
            backbone = registry.get_model_class(bb_name)(**bb_cfg)
        
        self.backbone = backbone
        self.reduce = reduce
        self.use_initial_embeddings = use_initial_embeddings
        
        # Energy prediction block (same as standard head)
        self.energy_block = EnergyBlock(
            backbone.sphere_channels_all, backbone.num_sphere_samples, backbone.act
        )
        
        # Weight prediction network - ONLY initialize if using weighted reduction
        if "weighted" in self.reduce:
            weight_input_dim = backbone.sphere_channels_all
            
            self.weight_nn = nn.Sequential(
                nn.Linear(weight_input_dim, weight_nn_hidden_dim),
                nn.SiLU(),
                nn.Linear(weight_nn_hidden_dim, weight_nn_hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(weight_nn_hidden_dim // 2, 1),
                nn.Sigmoid()  # Use Sigmoid for bounded weights [0, 1]
            )
            
            # Optional: additional weight network from initial embeddings (w-init in PhAST)
            if use_initial_embeddings:
                self.weight_nn_init = nn.Sequential(
                    nn.Linear(weight_input_dim, weight_nn_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(weight_nn_hidden_dim, 1),
                    nn.Sigmoid()  # Bounded weights for stability
                )
            else:
                self.weight_nn_init = None
        else:
            self.weight_nn = None
            self.weight_nn_init = None
    
    def forward(
        self, data: Batch, emb: dict[str, torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        # Get embeddings from backbone
        if emb is None:
            if hasattr(self.backbone, "forward_embeddings"):
                emb = self.backbone.forward_embeddings(data)
            else:
                raise AttributeError(
                    "backbone does not have forward_embeddings; cannot build sphere_values for head"
                )
        
        sphere_values = emb["sphere_values"]  # [N_atoms * num_samples, channels]
        num_sphere_samples = self.backbone.num_sphere_samples
        
        # Compute per-atom energy contributions (EnergyBlock aggregates internally)
        node_energy = self.energy_block(sphere_values)  # [N_atoms, 1]
        
        # --- Weighted Logic ---
        if self.weight_nn is not None:
            # Compute per-sample weights, then aggregate like EnergyBlock does
            node_weights_samples = self.weight_nn(sphere_values)  # [N_atoms * num_samples, 1]
            
            # Optional: add weights from initial embeddings
            if self.weight_nn_init is not None:
                node_weights_init_samples = self.weight_nn_init(sphere_values)
                node_weights_samples = node_weights_samples + node_weights_init_samples
            
            # Aggregate weights across sphere samples
            node_weights_samples = node_weights_samples.view(-1, num_sphere_samples, 1)
            node_weights = torch.sum(node_weights_samples, dim=1) / num_sphere_samples  # [N_atoms, 1]
            
            # Apply weights
            weighted_node_energy = node_energy * node_weights
        else:
            # Standard Sum/Mean logic (weights = 1.0)
            weighted_node_energy = node_energy
            node_weights = torch.ones_like(node_energy)
        
        # Aggregate per system
        num_systems = len(data.natoms)
        energy = torch.zeros(num_systems, device=data.pos.device)
        
        # Sum weighted contributions per graph
        energy.index_add_(0, data.batch, weighted_node_energy.view(-1))
        
        # Normalize by sum of weights per graph (important for stability)
        if self.reduce == "weighted_sum_normalized" or self.reduce == "mean":
            weight_sums = torch.zeros(num_systems, device=data.pos.device)
            weight_sums.index_add_(0, data.batch, node_weights.view(-1))
            # Avoid division by zero
            weight_sums = torch.clamp(weight_sums, min=1e-6)
            energy = energy / weight_sums
        elif self.reduce == "weighted_sum" or self.reduce == "sum":
            # Just return the sum, but maybe scaling is needed if weights are small?
            # PhAST uses sum logic.
            pass
        else:
            raise ValueError(f"Unknown reduce method: {self.reduce}")
        
        return {"energy": energy}



@registry.register_model("escn_force_head")
class eSCNForceHead(nn.Module, HeadInterface):
    def __init__(self, backbone, **kwargs):
        super().__init__()
        backbone.force_block = None
        self.force_block = ForceBlock(
            backbone.sphere_channels_all, backbone.num_sphere_samples, backbone.act
        )

    def forward(
        self, data: Batch, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {"forces": self.force_block(emb["sphere_values"], emb["sphere_points"])}


class LayerBlock(torch.nn.Module):
    """
    Layer block: Perform one layer (message passing and aggregation) of the GNN

    Args:
        layer_idx (int):            Layer number
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        layer_idx: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        distance_expansion,
        max_num_elements: int,
        SO3_grid: SO3_Grid,
        act,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.act = act
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.sphere_channels = sphere_channels
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.SO3_grid = SO3_grid

        # Message block
        self.message_block = MessageBlock(
            self.layer_idx,
            self.sphere_channels,
            hidden_channels,
            edge_channels,
            self.lmax_list,
            self.mmax_list,
            distance_expansion,
            max_num_elements,
            self.SO3_grid,
            self.act,
        )

        # Non-linear point-wise comvolution for the aggregated messages
        self.fc1_sphere = nn.Linear(
            2 * self.sphere_channels_all, self.sphere_channels_all, bias=False
        )

        self.fc2_sphere = nn.Linear(
            self.sphere_channels_all, self.sphere_channels_all, bias=False
        )

        self.fc3_sphere = nn.Linear(
            self.sphere_channels_all, self.sphere_channels_all, bias=False
        )

    def forward(
        self,
        x,
        atomic_numbers,
        edge_distance,
        edge_index,
        SO3_edge_rot,
        mappingReduced,
    ):
        # Compute messages by performing message block
        x_message = self.message_block(
            x,
            atomic_numbers,
            edge_distance,
            edge_index,
            SO3_edge_rot,
            mappingReduced,
        )

        # Compute point-wise spherical non-linearity on aggregated messages
        max_lmax = max(self.lmax_list)

        # Project to grid
        x_grid_message = x_message.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid = x.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid = torch.cat([x_grid, x_grid_message], dim=3)

        # Perform point-wise convolution
        x_grid = self.act(self.fc1_sphere(x_grid))
        x_grid = self.act(self.fc2_sphere(x_grid))
        x_grid = self.fc3_sphere(x_grid)

        # Project back to spherical harmonic coefficients
        x_message._from_grid(x_grid, self.SO3_grid, lmax=max_lmax)

        # Return aggregated messages
        return x_message


class MessageBlock(torch.nn.Module):
    """
    Message block: Perform message passing

    Args:
        layer_idx (int):            Layer number
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        layer_idx: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        distance_expansion,
        max_num_elements: int,
        SO3_grid: SO3_Grid,
        act,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.act = act
        self.hidden_channels = hidden_channels
        self.sphere_channels = sphere_channels
        self.SO3_grid = SO3_grid
        self.num_resolutions = len(lmax_list)
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.edge_channels = edge_channels

        # Create edge scalar (invariant to rotations) features
        self.edge_block = EdgeBlock(
            self.edge_channels,
            distance_expansion,
            max_num_elements,
            self.act,
        )

        # Create SO(2) convolution blocks
        self.so2_block_source = SO2Block(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax_list,
            self.mmax_list,
            self.act,
        )
        self.so2_block_target = SO2Block(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax_list,
            self.mmax_list,
            self.act,
        )

    def forward(
        self,
        x,
        atomic_numbers,
        edge_distance,
        edge_index,
        SO3_edge_rot,
        mappingReduced,
    ):
        ###############################################################
        # Compute messages
        ###############################################################

        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        x_edge = self.edge_block(
            edge_distance,
            atomic_numbers[edge_index[0]],  # Source atom atomic number
            atomic_numbers[edge_index[1]],  # Target atom atomic number
        )

        # Copy embeddings for each edge's source and target nodes
        x_source = x.clone()
        x_target = x.clone()
        x_source._expand_edge(edge_index[0, :])
        x_target._expand_edge(edge_index[1, :])

        # Rotate the irreps to align with the edge
        x_source._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)
        x_target._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)

        # Compute messages
        x_source = self.so2_block_source(x_source, x_edge, mappingReduced)
        x_target = self.so2_block_target(x_target, x_edge, mappingReduced)

        # Add together the source and target results
        x_target.embedding = x_source.embedding + x_target.embedding

        # Point-wise spherical non-linearity
        x_target._grid_act(self.SO3_grid, self.act, mappingReduced)

        # Rotate back the irreps
        x_target._rotate_inv(SO3_edge_rot, mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        x_target._reduce_edge(edge_index[1], len(x.embedding))

        return x_target


class SO2Block(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        act,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions: int = len(lmax_list)
        self.act = act

        num_channels_m0 = 0
        for i in range(self.num_resolutions):
            num_coefficents = self.lmax_list[i] + 1
            num_channels_m0 = num_channels_m0 + num_coefficents * self.sphere_channels

        # SO(2) convolution for m=0
        self.fc1_dist0 = nn.Linear(edge_channels, self.hidden_channels)
        self.fc1_m0 = nn.Linear(num_channels_m0, self.hidden_channels, bias=False)
        self.fc2_m0 = nn.Linear(self.hidden_channels, num_channels_m0, bias=False)

        # SO(2) convolution for non-zero m
        self.so2_conv = nn.ModuleList()
        for m in range(1, max(self.mmax_list) + 1):
            so2_conv = SO2Conv(
                m,
                self.sphere_channels,
                self.hidden_channels,
                edge_channels,
                self.lmax_list,
                self.mmax_list,
                self.act,
            )
            self.so2_conv.append(so2_conv)

    def forward(
        self,
        x,
        x_edge,
        mappingReduced,
    ):
        num_edges = len(x_edge)

        # Reshape the spherical harmonics based on m (order)
        x._m_primary(mappingReduced)

        # Compute m=0 coefficients separately since they only have real values (no imaginary)

        # Compute edge scalar features for m=0
        x_edge_0 = self.act(self.fc1_dist0(x_edge))

        x_0 = x.embedding[:, 0 : mappingReduced.m_size[0]].contiguous()
        x_0 = x_0.view(num_edges, -1)

        x_0 = self.fc1_m0(x_0)
        x_0 = x_0 * x_edge_0
        x_0 = self.fc2_m0(x_0)
        x_0 = x_0.view(num_edges, -1, x.num_channels)

        # Update the m=0 coefficients
        x.embedding[:, 0 : mappingReduced.m_size[0]] = x_0

        # Compute the values for the m > 0 coefficients
        offset = mappingReduced.m_size[0]
        for m in range(1, max(self.mmax_list) + 1):
            # Get the m order coefficients
            x_m = x.embedding[
                :, offset : offset + 2 * mappingReduced.m_size[m]
            ].contiguous()
            x_m = x_m.view(num_edges, 2, -1)
            # Perform SO(2) convolution
            x_m = self.so2_conv[m - 1](x_m, x_edge)
            x_m = x_m.view(num_edges, -1, x.num_channels)
            x.embedding[:, offset : offset + 2 * mappingReduced.m_size[m]] = x_m

            offset = offset + 2 * mappingReduced.m_size[m]

        # Reshape the spherical harmonics based on l (degree)
        x._l_primary(mappingReduced)

        return x


class SO2Conv(torch.nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        m: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        act,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.sphere_channels = sphere_channels
        self.num_resolutions: int = len(self.lmax_list)
        self.m = m
        self.act = act

        num_channels = 0
        for i in range(self.num_resolutions):
            num_coefficents = 0
            if self.mmax_list[i] >= m:
                num_coefficents = self.lmax_list[i] - m + 1

            num_channels = num_channels + num_coefficents * self.sphere_channels

        assert num_channels > 0

        # Embedding function of the distance
        self.fc1_dist = nn.Linear(edge_channels, 2 * self.hidden_channels)

        # Real weights of SO(2) convolution
        self.fc1_r = nn.Linear(num_channels, self.hidden_channels, bias=False)
        self.fc2_r = nn.Linear(self.hidden_channels, num_channels, bias=False)

        # Imaginary weights of SO(2) convolution
        self.fc1_i = nn.Linear(num_channels, self.hidden_channels, bias=False)
        self.fc2_i = nn.Linear(self.hidden_channels, num_channels, bias=False)

    def forward(self, x_m, x_edge) -> torch.Tensor:
        # Compute edge scalar features
        x_edge = self.act(self.fc1_dist(x_edge))
        x_edge = x_edge.view(-1, 2, self.hidden_channels)

        # Perform the complex weight multiplication
        x_r = self.fc1_r(x_m)
        x_r = x_r * x_edge[:, 0:1, :]
        x_r = self.fc2_r(x_r)

        x_i = self.fc1_i(x_m)
        x_i = x_i * x_edge[:, 1:2, :]
        x_i = self.fc2_i(x_i)

        x_m_r = x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r[:, 1] + x_i[:, 0]

        return torch.stack((x_m_r, x_m_i), dim=1).contiguous()


class EdgeBlock(torch.nn.Module):
    """
    Edge Block: Compute invariant edge representation from edge diatances and atomic numbers

    Args:
        edge_channels (int):        Size of invariant edge embedding
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        edge_channels,
        distance_expansion,
        max_num_elements,
        act,
    ) -> None:
        super().__init__()
        self.in_channels = distance_expansion.num_output
        self.distance_expansion = distance_expansion
        self.act = act
        self.edge_channels = edge_channels
        self.max_num_elements = max_num_elements

        # Embedding function of the distance
        self.fc1_dist = nn.Linear(self.in_channels, self.edge_channels)

        # Embedding function of the atomic numbers
        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        # Embedding function of the edge
        self.fc1_edge_attr = nn.Linear(
            self.edge_channels,
            self.edge_channels,
        )

    def forward(self, edge_distance, source_element, target_element):
        # Compute distance embedding
        x_dist = self.distance_expansion(edge_distance)
        x_dist = self.fc1_dist(x_dist)

        # Compute atomic number embeddings
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)

        # Compute invariant edge embedding
        x_edge = self.act(source_embedding + target_embedding + x_dist)
        return self.act(self.fc1_edge_attr(x_edge))


class EnergyBlock(torch.nn.Module):
    """
    Energy Block: Output block computing the energy

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        num_channels: int,
        num_sphere_samples: int,
        act,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_sphere_samples = num_sphere_samples
        self.act = act

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

    def forward(self, x_pt) -> torch.Tensor:
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        return torch.sum(x_pt, dim=1) / self.num_sphere_samples


class ForceBlock(torch.nn.Module):
    """
    Force Block: Output block computing the per atom forces

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        num_channels: int,
        num_sphere_samples: int,
        act,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_sphere_samples = num_sphere_samples
        self.act = act

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

    def forward(self, x_pt, sphere_points) -> torch.Tensor:
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        forces = x_pt * sphere_points.view(1, self.num_sphere_samples, 3)
        return torch.sum(forces, dim=1) / self.num_sphere_samples
