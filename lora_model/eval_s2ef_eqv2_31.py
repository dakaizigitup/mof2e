"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import io
import os

import pytest
import requests
import torch
from ase.io import read
import multiprocessing
import pdb
import sys
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
import torch.distributed as dist
from torch.distributed.elastic.utils.distributed import get_free_port
from fairchem.core.common.gp_utils import setup_gp
from torch.nn.parallel.distributed import DistributedDataParallel
from fairchem.core.common.registry import registry
from fairchem.core.common.test_utils import (
    PGConfig,
    init_pg_and_rank_and_launch_test,
    spawn_multi_process,
)
from fairchem.core.common.utils import load_state_dict, setup_imports
from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.models.equiformer_v2.equiformer_v2_deprecated import EquiformerV2


@pytest.fixture(scope="class")
def load_data(request):
    atoms = read(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json"),
        index=0,
        format="json",
    )
    a2g = AtomsToGraphs(
        max_neigh=200,
        radius=6,
        r_edges=False,
        r_fixed=True,
    )
    data_list = a2g.convert_all([atoms])
    request.cls.data = data_list[0]


def _load_model():
    torch.manual_seed(4)
    setup_imports()

    # # download and load weights.
    # checkpoint_url = "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_31M_ec4_allmd.pt"

    # # load buffer into memory as a stream
    # # and then load it with torch.load
    # r = requests.get(checkpoint_url, stream=True)
    # r.raise_for_status()
    # checkpoint = torch.load(io.BytesIO(r.content), map_location=torch.device("cpu"))
    ckpt = "/data0/wfz/code/fairchem/configs/odac/s2ef/ckpt_ocp/eqv2_31M.pt"
    checkpoint = torch.load(ckpt, map_location=torch.device("cpu"))

    model = registry.get_model_class("equiformer_v2")(
        use_pbc=True,
        regress_forces=True,
        otf_graph=True,
        max_neighbors=20,
        max_radius=8.0,
        max_num_elements=100,
        num_layers=8,
        sphere_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=64,
        attn_value_channels=16,
        ffn_hidden_channels=128,
        norm_type="layer_norm_sh",
        lmax_list=[4],
        mmax_list=[2],
        grid_resolution=18,
        num_sphere_samples=128,
        edge_channels=128,
        use_atom_edge_embedding=True,
        distance_function="gaussian",
        num_distance_basis=512,
        attn_activation="silu",
        use_s2_act_attn=False,
        ffn_activation="silu",
        use_gate_act=False,
        use_grid_mlp=True,
        alpha_drop=0.1,
        drop_path_rate=0.1,
        proj_drop=0.0,
        weight_init="uniform",
    )

    new_dict = {k[len("module.") * 2 :]: v for k, v in checkpoint["state_dict"].items()}
    load_state_dict(model, new_dict)

    # Precision errors between mac vs. linux compound with multiple layers,
    # so we explicitly set the number of layers to 1 (instead of all 8).
    # The other alternative is to have different snapshots for mac vs. linux.
    # model.num_layers = 1
    return model


@pytest.fixture(scope="class")
def load_model(request):
    request.cls.model = _load_model()


def _runner(data):
    # serializing the model through python multiprocess results in precision errors, so we get a fresh model here
    model = _load_model()
    ddp_model = DistributedDataParallel(model)
    outputs = ddp_model(data_list_collater([data]))
    return {k: v.detach() for k, v in outputs.items()}


@pytest.mark.usefixtures("load_data")
@pytest.mark.usefixtures("load_model")
class TestEquiformerV2:
    def test_energy_force_shape(self, snapshot):
        # Recreate the Data object to only keep the necessary features.
        data = self.data
        model = copy.deepcopy(self.model)

        # Pass it through the model.
        outputs = model(data_list_collater([data]))
        print(outputs)
        energy, forces = outputs["energy"], outputs["forces"]

        assert snapshot == energy.shape
        assert snapshot == pytest.approx(energy.detach())

        assert snapshot == forces.shape
        assert snapshot == pytest.approx(forces.detach().mean(0))

    def test_ddp(self, snapshot):
        data_dist = self.data.clone().detach()
        config = PGConfig(backend="gloo", world_size=1, gp_group_size=1, use_gp=False)
        output = spawn_multi_process(
            config, _runner, init_pg_and_rank_and_launch_test, data_dist
        )
        assert len(output) == 1
        energy, forces = output[0]["energy"], output[0]["forces"]
        assert snapshot == energy.shape
        assert snapshot == pytest.approx(energy.detach())
        assert snapshot == forces.shape
        assert snapshot == pytest.approx(forces.detach().mean(0))

    def test_gp(self, snapshot):
        data_dist = self.data.clone().detach()
        config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
        output = spawn_multi_process(
            config, _runner, init_pg_and_rank_and_launch_test, data_dist
        )
        assert len(output) == 2
        energy, forces = output[0]["energy"], output[0]["forces"]
        assert snapshot == energy.shape
        assert snapshot == pytest.approx(energy.detach())
        assert snapshot == forces.shape
        assert snapshot == pytest.approx(forces.detach().mean(0))



class Equiformerv2Lora(EquiformerV2):
    def init():

    def 

    def forward(self, data):
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

        ###############################################################
        # Entering Graph Parallel Region
        # after this point, if using gp, then node, edge tensors are split
        # across the graph parallel ranks, some full tensors such as
        # atomic_numbers_full are required because we need to index into the
        # full graph when computing edge embeddings or reducing nodes from neighbors
        #
        # all tensors that do not have the suffix "_full" refer to the partial tensors.
        # if not using gp, the full values are equal to the partial values
        # ie: atomic_numbers_full == atomic_numbers
        ###############################################################

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, graph.edge_index, graph.edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)



        ###############################################################
        # Adding new MOF embedding
        ###############################################################




        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding 创建空的等变嵌入
        x = SO3_Embedding(
            len(atomic_numbers),
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        graph.edge_distance = self.distance_expansion(graph.edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = graph.atomic_numbers_full[
                graph.edge_index[0]
            ]  # Source atom atomic number
            target_element = graph.atomic_numbers_full[
                graph.edge_index[1]
            ]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            graph.edge_distance = torch.cat(
                (graph.edge_distance, source_embedding, target_embedding), dim=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            graph.atomic_numbers_full,
            graph.edge_distance,
            graph.edge_index,
            len(atomic_numbers),
            graph.node_offset,
        )
        x.embedding = x.embedding + edge_degree.embedding #包含球谐系数的等变特征张量

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                graph.atomic_numbers_full,
                graph.edge_distance,
                graph.edge_index,
                batch=data_batch,  # for GraphDropPath
                node_offset=graph.node_offset,
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x)
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        if gp_utils.initialized():
            node_energy = gp_utils.gather_from_model_parallel_region(node_energy, dim=0)
        energy = torch.zeros(
            len(data.natoms),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )
        energy.index_add_(0, graph.batch_full, node_energy.view(-1))
        energy = energy / self.avg_num_nodes

        # Add the per-atom linear references to the energy.
        if self.use_energy_lin_ref and self.load_energy_lin_ref:
            # During training, target E = (E_DFT - E_ref - E_mean) / E_std, and
            # during inference, \hat{E_DFT} = \hat{E} * E_std + E_ref + E_mean
            # where
            #
            # E_DFT = raw DFT energy,
            # E_ref = reference energy,
            # E_mean = normalizer mean,
            # E_std = normalizer std,
            # \hat{E} = predicted energy,
            # \hat{E_DFT} = predicted DFT energy.
            #
            # We can also write this as
            # \hat{E_DFT} = E_std * (\hat{E} + E_ref / E_std) + E_mean,
            # which is why we save E_ref / E_std as the linear reference.
            with torch.autocast("cuda", enabled=False):
                energy = energy.to(self.energy_lin_ref.dtype).index_add(
                    0,
                    graph.batch_full,
                    self.energy_lin_ref[graph.atomic_numbers_full],
                )

        outputs = {"energy": energy}
        ###############################################################
        # Force estimation
        ###############################################################
        if self.regress_forces:
            forces = self.force_block(
                x,
                graph.atomic_numbers_full,
                graph.edge_distance,
                graph.edge_index,
                node_offset=graph.node_offset,
            )
            forces = forces.embedding.narrow(1, 1, 3)
            forces = forces.view(-1, 3).contiguous()
            if gp_utils.initialized():
                forces = gp_utils.gather_from_model_parallel_region(forces, dim=0)
            outputs["forces"] = forces

        return outputs


