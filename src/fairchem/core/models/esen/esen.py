"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import logging

import torch
import torch.nn as nn

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import GraphModelMixin, HeadInterface

from .common.rotation import (
    init_edge_rot_mat,
    rotation_to_wigner,
)
from .common.so3 import (
    CoefficientMapping,
    SO3_Grid,
)
from .esen_block import eSEN_Block
from .nn.embedding import EdgeDegreeEmbedding
from .nn.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .nn.radial import EnvelopedBesselBasis, GaussianSmearing
from .nn.so3_layers import SO3_Linear


@registry.register_model("esen_backbone")
class eSEN_Backbone(nn.Module, GraphModelMixin):
    def __init__(
        self,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        otf_graph: bool = False,
        max_neighbors: int = 300,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        cutoff: float = 5.0,
        edge_channels: int = 128,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        direct_forces: bool = True,
        regress_forces: bool = True,
        regress_stress: bool = False,
        # escnmd specific
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "s2",
        mlp_type: str = "grid",
        use_envelope: bool = False,
        activation_checkpointing: bool = False,
        # ---- New feature injection params from eSCN ----
        base_nm_dim: int = 2,
        use_atom_extra_features: bool = False,
        atom_extra_dim: int = 16, # Fixed to 16 in eSCN
        use_mof_global_features: bool = False,
        mof_global_excel_path: str | None = None,
        mof_global_dim: int = 32, # Fixed to 32 in eSCN
        adapter_hidden_dim: int = 128,
        adapter_dropout: float = 0.0,
        adapter_zero_init: bool = True,
    ):
        super().__init__()

        self.max_num_elements = max_num_elements
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.grid_resolution = grid_resolution

        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.regress_stress = regress_stress

        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.enforce_max_neighbors_strictly = False
        self.activation_checkpointing = activation_checkpointing

        self.mlp_type = mlp_type
        self.use_envelope = use_envelope

        # ---- New feature injection logic from eSCN ----
        self.use_atom_extra_features = bool(use_atom_extra_features)
        self.atom_extra_dim = 16 # Fixed value from eSCN
        self.use_mof_global_features = bool(use_mof_global_features)
        self.mof_global_excel_path = mof_global_excel_path
        self.mof_global_dim = 32 # Fixed value from eSCN
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.adapter_dropout = float(adapter_dropout)
        self.adapter_zero_init = bool(adapter_zero_init)

        self.atom_encoder = None
        self.mof_global_encoder = None

        if self.use_atom_extra_features:
            try:
                from fairchem.core.models.equiformer_v2.atomic_emb import AtomPropertyEncoder

                self.atom_encoder = AtomPropertyEncoder(
                    max_Z=self.max_num_elements, out_dim=self.atom_extra_dim
                )
            except Exception as e:
                logging.warning(f"Failed to init AtomPropertyEncoder: {e}")
                self.atom_encoder = None

        if self.use_mof_global_features and self.mof_global_excel_path:
            try:
                from fairchem.core.models.equiformer_v2.global_emb import MOFGlobalPropertyEncoder

                self.mof_global_encoder = MOFGlobalPropertyEncoder(
                    excel_path=self.mof_global_excel_path,
                    global_dim=self.mof_global_dim,
                    hidden_dim=max(256, self.mof_global_dim * 2),
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            except Exception as e:
                logging.warning(f"Failed to init MOFGlobalPropertyEncoder: {e}")
                self.mof_global_encoder = None

        self.base_nm_dim = base_nm_dim
        self.context_dim = self.base_nm_dim
        if self.use_mof_global_features and self.mof_global_encoder is not None:
            self.context_dim += self.mof_global_dim
        if self.use_atom_extra_features and self.atom_encoder is not None:
            self.context_dim += self.atom_extra_dim

        self.fused_dim = self.sphere_channels + self.context_dim

        self.fused_ln = None
        self.fused_mlp = None
        if self.context_dim > 0:
            self.fused_ln = nn.LayerNorm(self.fused_dim)
            self.fused_mlp = nn.Sequential(
                nn.Linear(self.fused_dim, self.adapter_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.adapter_dropout),
                nn.Linear(self.adapter_hidden_dim, self.sphere_channels),
            )
            if self.adapter_zero_init:
                for m in reversed(self.fused_mlp):
                    if isinstance(m, nn.Linear):
                        nn.init.zeros_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                        break

        # rotation utils
        Jd_list = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
        for l in range(self.lmax + 1):
            self.register_buffer(f"Jd_{l}", Jd_list[l])
        self.sph_feature_size = int((self.lmax + 1) ** 2)
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)

        # lmax_lmax for node, lmax_mmax for edge
        self.SO3_grid = nn.ModuleDict()
        self.SO3_grid["lmax_lmax"] = SO3_Grid(
            self.lmax, self.lmax, resolution=grid_resolution, rescale=True
        )
        self.SO3_grid["lmax_mmax"] = SO3_Grid(
            self.lmax, self.mmax, resolution=grid_resolution, rescale=True
        )

        # atom embedding
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels
        )

        # edge distance embedding
        self.cutoff = cutoff
        self.edge_channels = edge_channels
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                self.num_distance_basis,
                2.0,
            )
        elif self.distance_function == "bessel":
            self.distance_expansion = EnvelopedBesselBasis(
                num_radial=self.num_distance_basis,
                cutoff=cutoff,
            )
            self.distance_expansion.offset = [self.cutoff]
            self.distance_expansion.num_output = self.num_distance_basis
        else:
            raise ValueError("Unknown distance function")

        # equivariant initial embedding
        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.edge_channels_list = [
            self.num_distance_basis + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            max_num_elements=self.max_num_elements,
            edge_channels_list=self.edge_channels_list,
            rescale_factor=5.0,
            cutoff=self.cutoff,
            mappingReduced=self.mappingReduced,
            out_mask=self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
                self.lmax, self.mmax
            ),
            use_envelope=use_envelope,
        )

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.act_type = act_type

        # Initialize the blocks for each layer
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            block = eSEN_Block(
                self.sphere_channels,
                self.hidden_channels,
                self.lmax,
                self.mmax,
                self.mappingReduced,
                self.SO3_grid,
                self.edge_channels_list,
                self.cutoff,
                self.norm_type,
                self.act_type,
                self.mlp_type,
                self.use_envelope,
            )
            self.blocks.append(block)

        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels,
        )

    def get_rotmat_and_wigner(self, edge_distance_vecs):
        edge_rot_mat = init_edge_rot_mat(
            edge_distance_vecs, rot_clip=(not self.direct_forces)
        )

        Jd_buffers = [
            getattr(self, f"Jd_{l}").type(edge_rot_mat.dtype)
            for l in range(self.lmax + 1)
        ]

        wigner = rotation_to_wigner(
            edge_rot_mat,
            0,
            self.lmax,
            Jd_buffers,
            rot_clip=(not self.direct_forces),
        )
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

        return edge_rot_mat, wigner, wigner_inv

    def generate_graph(self, *args, **kwargs):
        graph = super().generate_graph(*args, **kwargs)
        return {
            "edge_index": graph.edge_index,
            "edge_distance": graph.edge_distance,
            "edge_distance_vec": graph.edge_distance_vec,
            "cell_offsets": graph.cell_offsets,
            "offset_distances": None,
            "neighbors": None,
            "node_offset": 0,
            "batch_full": graph.batch_full,
            "atomic_numbers_full": graph.atomic_numbers_full,
        }

    @conditional_grad(torch.enable_grad())
    def forward(self, data_dict) -> dict[str, torch.Tensor]:
        ###############################################################
        # gradient-based forces/stress
        ###############################################################
        data_dict["atomic_numbers"] = data_dict["atomic_numbers"].long()

        displacement = None
        orig_cell = None
        if self.regress_stress and not self.direct_forces:
            displacement = torch.zeros(
                (3, 3),
                dtype=data_dict["pos"].dtype,
                device=data_dict["pos"].device,
            )
            # num_batch = data_dict["num_graphs"]
            num_batch = data_dict.get("num_graphs", len(data_dict["natoms"]))
            displacement = displacement.view(-1, 3, 3).expand(num_batch, 3, 3)
            displacement.requires_grad = True
            symmetric_displacement = 0.5 * (
                displacement + displacement.transpose(-1, -2)
            )

            data_dict["pos"].requires_grad = True
            data_dict["pos"] = data_dict["pos"] + torch.bmm(
                data_dict["pos"].unsqueeze(-2),
                torch.index_select(symmetric_displacement, 0, data_dict["batch"]),
            ).squeeze(-2)

            orig_cell = data_dict["cell"]
            data_dict["cell"] = data_dict["cell"] + torch.bmm(
                data_dict["cell"], symmetric_displacement
            )

        if not self.regress_stress and self.regress_forces and not self.direct_forces:
            data_dict["pos"].requires_grad = True

        if self.otf_graph:
            graph_dict = self.generate_graph(data_dict)
        else:
            cell_per_edge = data_dict["cell"].repeat_interleave(
                data_dict["nedges"], dim=0
            )
            shifts = torch.einsum(
                "ij,ijk->ik",
                data_dict["cell_offsets"].to(cell_per_edge.dtype),
                cell_per_edge,
            )
            edge_distance_vec = (
                data_dict["pos"][data_dict["edge_index"][0]]
                - data_dict["pos"][data_dict["edge_index"][1]]
                + shifts
            )
            # pylint: disable=E1102
            edge_distance = torch.linalg.norm(edge_distance_vec, dim=-1, keepdim=False)
            graph_dict = {
                "atomic_numbers_full": data_dict["atomic_numbers_full"],
                "batch_full": data_dict["batch_full"],
                "edge_index": data_dict["edge_index"],
                "edge_distance": edge_distance,
                "edge_distance_vec": edge_distance_vec,
                "node_offset": 0,
            }

        _, wigner, wigner_inv = self.get_rotmat_and_wigner(
            graph_dict["edge_distance_vec"]
        )

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        x_message = torch.zeros(
            data_dict["pos"].shape[0],
            self.sph_feature_size,
            self.sphere_channels,
            device=data_dict["pos"].device,
            dtype=data_dict["pos"].dtype,
        )
        x_message[:, 0, :] = self.sphere_embedding(data_dict["atomic_numbers"])

        # ---- Feature injection logic from eSCN ----
        if self.fused_mlp is not None:
            context_features = []
            num_atoms = data_dict["pos"].shape[0]
            device = data_dict["pos"].device

            # 1. nm(condition) features
            if "condition" in data_dict and data_dict["condition"] is not None:
                cond = data_dict["condition"]
                if isinstance(cond, torch.Tensor):
                    cond = cond.to(device=device, dtype=torch.float32)
                    if cond.dim() == 1:
                        cond = cond.view(1, -1)
                else:
                    cond = torch.tensor(cond, device=device, dtype=torch.float32).view(1, -1)

                if cond.size(0) == 1 and cond.size(1) > self.base_nm_dim:
                    num_conditions = cond.size(1) // self.base_nm_dim
                    if cond.size(1) % self.base_nm_dim == 0 and num_conditions == len(data_dict["natoms"]):
                        cond = cond.view(num_conditions, self.base_nm_dim)

                if cond.size(-1) != self.base_nm_dim:
                    raise ValueError(f"data.condition last dim must be {self.base_nm_dim}, got {tuple(cond.shape)}")

                if cond.size(0) == 1 and len(data_dict["natoms"]) > 1:
                    cond = cond.expand(len(data_dict["natoms"]), -1)

                batch_indices = data_dict["batch"] if "batch" in data_dict else torch.zeros(num_atoms, dtype=torch.long, device=device)
                nm_per_node = cond[batch_indices]
                context_features.append(nm_per_node)
            else:
                context_features.append(torch.zeros(num_atoms, self.base_nm_dim, device=device, dtype=torch.float32))

            # 2. Global MOF features
            if self.use_mof_global_features and self.mof_global_encoder is not None:
                mof_embeddings = []
                if "name" in data_dict and data_dict["name"] is not None:
                    names = data_dict["name"]
                    if isinstance(names, (list, tuple)):
                        for name in names:
                            mof_embeddings.append(self.mof_global_encoder(str(name)))
                    else:
                        mof_embeddings = [self.mof_global_encoder(str(names))] * len(data_dict["natoms"])

                if mof_embeddings:
                    global_emb = torch.cat(mof_embeddings, dim=0).to(device=device)
                    global_emb = torch.nan_to_num(global_emb, nan=0.0, posinf=0.0, neginf=0.0)
                    batch_indices = data_dict["batch"] if "batch" in data_dict else torch.zeros(num_atoms, dtype=torch.long, device=device)
                    global_per_node = global_emb[batch_indices]
                    context_features.append(global_per_node)

            # 3. Atomic-level features
            if self.use_atom_extra_features and self.atom_encoder is not None:
                atom_feat = self.atom_encoder(data_dict["atomic_numbers"]).to(device=device)
                context_features.append(atom_feat)

            # 4. Fuse features and apply residual update
            l0_features = x_message[:, 0, :].clone()
            context = torch.cat(context_features, dim=-1)
            fused_input = torch.cat([l0_features, context], dim=-1)
            fused_input = self.fused_ln(fused_input)
            delta = self.fused_mlp(fused_input)
            x_message[:, 0, :] = l0_features + delta

        # edge degree embedding
        edge_distance_embedding = self.distance_expansion(graph_dict["edge_distance"])
        source_embedding = self.source_embedding(
            data_dict["atomic_numbers"][graph_dict["edge_index"][0]]
        )
        target_embedding = self.target_embedding(
            data_dict["atomic_numbers"][graph_dict["edge_index"][1]]
        )
        x_edge = torch.cat(
            (edge_distance_embedding, source_embedding, target_embedding), dim=1
        )
        x_message = self.edge_degree_embedding(
            x_message,
            x_edge,
            graph_dict["edge_distance"],
            graph_dict["edge_index"],
            wigner_inv,
        )

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        if graph_dict["edge_index"].shape[1] != 0:
            for i in range(self.num_layers):
                if self.activation_checkpointing:
                    x_message = torch.utils.checkpoint.checkpoint(
                        self.blocks[i],
                        x_message,
                        x_edge,
                        graph_dict["edge_distance"],
                        graph_dict["edge_index"],
                        wigner,
                        wigner_inv,
                        graph_dict["node_offset"],
                        use_reentrant=False,
                    )
                else:
                    x_message = self.blocks[i](
                        x_message,
                        x_edge,
                        graph_dict["edge_distance"],
                        graph_dict["edge_index"],
                        wigner,
                        wigner_inv,
                        node_offset=graph_dict["node_offset"],
                    )

        # Final layer norm
        x_message = self.norm(x_message)

        out = {
            "node_embedding": x_message,
            "displacement": displacement,
            "orig_cell": orig_cell,
        }
        out.update(graph_dict)

        return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_Linear,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, (torch.nn.Linear, SO3_Linear))
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)


@registry.register_model("esen_mlp_efs_head")
class MLP_EFS_Head(nn.Module, HeadInterface):
    def __init__(self, backbone):
        super().__init__()
        backbone.energy_block = None
        backbone.force_block = None
        self.regress_stress = backbone.regress_stress
        self.regress_forces = backbone.regress_forces

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

        backbone.direct_forces = False

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_key = "energy"
        forces_key = "forces"
        stress_key = "stress"

        outputs = {}

        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze()
        ).view(-1, 1, 1)

        energy = torch.zeros(
            len(data["natoms"]), device=data["pos"].device, dtype=node_energy.dtype
        )
        energy.index_add_(0, data["batch"], node_energy.view(-1))
        outputs[energy_key] = energy

        if self.regress_stress:
            grads = torch.autograd.grad(
                [energy.sum()],
                [data["pos"], emb["displacement"]],
                create_graph=self.training,
            )
            forces = torch.neg(grads[0])
            virial = grads[1].view(-1, 3, 3)
            volume = torch.det(data["cell"]).abs().unsqueeze(-1)
            stress = virial / volume.view(-1, 1, 1)
            virial = torch.neg(virial)
            outputs[forces_key] = forces
            outputs[stress_key] = stress.view(-1, 9)
            data["cell"] = emb["orig_cell"]
        elif self.regress_forces:
            forces = (
                -1
                * torch.autograd.grad(
                    energy.sum(), data["pos"], create_graph=self.training
                )[0]
            )
            outputs[forces_key] = forces
        return outputs


@registry.register_model("esen_mlp_energy_head")
class MLP_Energy_Head(nn.Module, HeadInterface):
    def __init__(self, backbone, reduce: str = "sum"):
        super().__init__()
        self.reduce = reduce

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze()
        ).view(-1, 1, 1)

        energy = torch.zeros(
            len(data_dict["natoms"]),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        energy.index_add_(0, data_dict["batch"], node_energy.view(-1))
        if self.reduce == "sum":
            return {"energy": energy}
        elif self.reduce == "mean":
            return {"energy": energy / data_dict["natoms"]}
        else:
            raise ValueError(
                f"reduce can only be sum or mean, user provided: {self.reduce}"
            )


@registry.register_model("esen_linear_force_head")
class Linear_Force_Head(nn.Module, HeadInterface):
    def __init__(self, backbone):
        super().__init__()
        self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
        forces = self.linear(emb["node_embedding"].narrow(1, 0, 4))
        forces = forces.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()
        return {"forces": forces}
