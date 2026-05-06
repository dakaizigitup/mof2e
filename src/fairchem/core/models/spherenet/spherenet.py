import logging

import torch
from torch import nn
from torch.nn import Linear, Embedding
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter
from math import sqrt

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import GraphModelMixin, HeadInterface
from fairchem.core.models.escn.escn import ConditionEncoder

from .geometric_computing import xyz_to_dat
from .features import dist_emb, angle_emb, torsion_emb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def swish(x):
    return x * torch.sigmoid(x)

class emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.reset_parameters()

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb

class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish, use_node_features=True, use_extra_node_feature=False):
        super(init, self).__init__()
        self.act = act
        self.use_node_features = use_node_features
        self.use_extra_node_feature = use_extra_node_feature
        if self.use_node_features:
            self.emb = Embedding(95, hidden_channels)
        else: # option to use no node features and a learned embedding vector for each node instead
            self.node_embedding = nn.Parameter(torch.empty((hidden_channels,)))
            nn.init.normal_(self.node_embedding)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        if self.use_extra_node_feature:
            self.lin = Linear(5 * hidden_channels, hidden_channels)
        else:
            self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_node_features:
            self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, node_feature, emb, i, j):
        rbf,_,_ = emb
        if self.use_node_features:
            x = self.emb(x)
        else:
            x = self.node_embedding[None, :].expand(x.shape[0], -1)
        if node_feature != None and self.use_extra_node_feature:
            x = torch.cat((x, node_feature), 1)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial,
        num_before_skip, num_after_skip, act=swish):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_t1.weight, scale=2.0)
        glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        x1,_ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init):
        super(update_v, self).__init__()
        self.act = act
        self.output_init = output_init
        self.out_channels = out_channels

        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == 'zeros':
            self.lin.weight.data.fill_(0)
        if self.output_init == 'GlorotOrthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, e, i, return_hidden=False):
        _, e2 = e
        v = scatter(e2, i, dim=0)
        v = self.lin_up(v)
        for lin in self.lins:
            v = self.act(lin(v))
        if return_hidden:
            return v
        v = self.lin(v)
        return v


class update_u(torch.nn.Module):
    def __init__(self):
        super(update_u, self).__init__()

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u


@registry.register_model("spherenet_global")
class SphereNet(torch.nn.Module, GraphModelMixin):
    r"""
         The spherical message passing neural network SphereNet from the `"Spherical Message Passing for 3D Molecular Graphs" <https://openreview.net/forum?id=givsRXsOt9r>`_ paper.
        
        Args:
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            int_emb_size (int, optional): Embedding size used for interaction triplets. (default: :obj:`64`)
            basis_emb_size_dist (int, optional): Embedding size used in the basis transformation of distance. (default: :obj:`8`)
            basis_emb_size_angle (int, optional): Embedding size used in the basis transformation of angle. (default: :obj:`8`)
            basis_emb_size_torsion (int, optional): Embedding size used in the basis transformation of torsion. (default: :obj:`8`)
            out_emb_channels (int, optional): Embedding size used for atoms in the output block. (default: :obj:`256`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`7`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
            envelop_exponent (int, optional): Shape of the smooth cutoff. (default: :obj:`5`)
            num_before_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`1`)
            num_after_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`2`)
            num_output_layers (int, optional): Number of linear layers for the output blocks. (default: :obj:`3`)
            act: (function, optional): The activation funtion. (default: :obj:`swish`)
            output_init: (str, optional): The initialization fot the output. It could be :obj:`GlorotOrthogonal` and :obj:`zeros`. (default: :obj:`GlorotOrthogonal`)
            
    """
    def __init__(
        self, energy_and_force=False, cutoff=5.0, num_layers=4,
        hidden_channels=128, out_channels=1, int_emb_size=64,
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        num_spherical=7, num_radial=6, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        act=swish, output_init='GlorotOrthogonal', use_node_features=True, use_extra_node_feature=False, extra_node_feature_dim=1,
        use_pbc=False, use_pbc_single=False, regress_forces=False, otf_graph=True, max_neighbors=50,
        base_nm_dim: int = 2, condition_hidden_dim: int = 64, nm_max_count: float = 15.0,
        use_atom_extra_features: bool = False, atom_extra_dim: int = 128, atom_encoder_type: str = "v2",
        use_mof_global_features: bool = False, mof_global_excel_path: str | None = None, mof_global_dim: int = 64,
        mof_global_encoder_type: str = "v2", mof_smi_ted_folder: str | None = None,
        mof_selfies_ted_path: str | None = None, mof_mhg_path: str | None = None,
        mof_smiles_cache_path: str | None = None, adapter_hidden_dim: int = 128,
        adapter_dropout: float = 0.0, adapter_zero_init: bool = True,
        global_inject_layer: int = 3, global_layer_norm: bool = False, freeze_first_n_layers: int = 0):
        super(SphereNet, self).__init__()

        self.cutoff = cutoff
        self.energy_and_force = energy_and_force or regress_forces
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.hidden_channels = hidden_channels
        self.out_emb_channels = out_emb_channels
        self.num_layers = num_layers
        self.use_extra_node_feature = use_extra_node_feature
        self.base_nm_dim = base_nm_dim
        self.use_atom_extra_features = bool(use_atom_extra_features)
        self.atom_extra_dim = int(atom_extra_dim)
        self.use_mof_global_features = bool(use_mof_global_features)
        self.mof_global_dim = int(mof_global_dim)
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.adapter_dropout = float(adapter_dropout)
        self.adapter_zero_init = bool(adapter_zero_init)
        self.global_inject_layer = int(global_inject_layer)
        self.global_layer_norm = bool(global_layer_norm)
        self.freeze_first_n_layers = int(freeze_first_n_layers)

        if use_extra_node_feature:
            self.extra_emb = Linear(extra_node_feature_dim, hidden_channels)

        self.init_e = init(num_radial, hidden_channels, act, use_node_features=use_node_features, use_extra_node_feature=use_extra_node_feature)
        self.init_v = update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init)
        self.init_u = update_u()
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)

        self.update_vs = torch.nn.ModuleList([
            update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init) for _ in range(num_layers)])
        self.update_us = torch.nn.ModuleList([update_u() for _ in range(num_layers)])
        self.output_lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial, num_before_skip, num_after_skip,act) for _ in range(num_layers)])

        def _zero_last(seq: nn.Sequential) -> None:
            for m in reversed(list(seq)):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    return

        self.condition_encoder = ConditionEncoder(
            input_dim=base_nm_dim,
            num_gaussians=20,
            out_dim=condition_hidden_dim,
            hidden_dim=adapter_hidden_dim,
            dropout=adapter_dropout,
            nm_max_count=nm_max_count,
        )
        self.nm_encoded_dim = condition_hidden_dim

        self.atom_encoder = None
        if self.use_atom_extra_features:
            try:
                if atom_encoder_type.lower() == "v2":
                    from fairchem.core.models.equiformer_v2.atomic_emb_v2 import AtomPropertyEncoderV2
                    self.atom_encoder = AtomPropertyEncoderV2(max_Z=95, out_dim=atom_extra_dim)
                else:
                    from fairchem.core.models.equiformer_v2.atomic_emb import AtomPropertyEncoder
                    self.atom_encoder = AtomPropertyEncoder(max_Z=95, out_dim=atom_extra_dim)
            except Exception as e:
                logging.warning(f"[SphereNetGlobal] AtomPropertyEncoder init failed: {e}")
                self.atom_encoder = None

        self.mof_global_encoder = None
        if self.use_mof_global_features and mof_global_excel_path:
            _dev = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                from fairchem.core.models.equiformer_v2.global_emb_v2 import MOFGlobalEncoderV2
                self.mof_global_encoder = MOFGlobalEncoderV2(
                    excel_path=mof_global_excel_path,
                    smi_ted_folder=mof_smi_ted_folder,
                    selfies_ted_path=mof_selfies_ted_path,
                    mhg_path=mof_mhg_path,
                    global_dim=mof_global_dim,
                    device=_dev,
                    cache_path=mof_smiles_cache_path,
                )
            except Exception as e:
                logging.warning(f"[SphereNetGlobal] MOFGlobalEncoderV2 init failed: {e}")

        self.nm_gamma_mlp = nn.Sequential(
            nn.Linear(self.nm_encoded_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, out_emb_channels),
        )
        self.nm_beta_mlp = nn.Sequential(
            nn.Linear(self.nm_encoded_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, out_emb_channels),
        )
        if adapter_zero_init:
            _zero_last(self.nm_gamma_mlp)
            _zero_last(self.nm_beta_mlp)

        self.atom_gamma_mlp = None
        self.atom_beta_mlp = None
        if self.atom_encoder is not None:
            self.atom_gamma_mlp = nn.Sequential(
                nn.Linear(atom_extra_dim, adapter_hidden_dim),
                nn.GELU(),
                nn.Linear(adapter_hidden_dim, out_emb_channels),
            )
            self.atom_beta_mlp = nn.Sequential(
                nn.Linear(atom_extra_dim, adapter_hidden_dim),
                nn.GELU(),
                nn.Linear(adapter_hidden_dim, out_emb_channels),
            )
            if adapter_zero_init:
                _zero_last(self.atom_gamma_mlp)
                _zero_last(self.atom_beta_mlp)

        self.global_input_norm = nn.LayerNorm(mof_global_dim) if (self.use_mof_global_features and self.global_layer_norm) else None
        self.global_gamma_mlp = None
        self.global_beta_mlp = None
        if self.use_mof_global_features:
            self.global_gamma_mlp = nn.Sequential(
                nn.Linear(mof_global_dim, adapter_hidden_dim),
                nn.LayerNorm(adapter_hidden_dim) if self.global_layer_norm else nn.Identity(),
                nn.GELU(),
                nn.Linear(adapter_hidden_dim, out_emb_channels),
            )
            self.global_beta_mlp = nn.Sequential(
                nn.Linear(mof_global_dim, adapter_hidden_dim),
                nn.LayerNorm(adapter_hidden_dim) if self.global_layer_norm else nn.Identity(),
                nn.GELU(),
                nn.Linear(adapter_hidden_dim, out_emb_channels),
            )
            if adapter_zero_init:
                _zero_last(self.global_gamma_mlp)
                _zero_last(self.global_beta_mlp)

        if freeze_first_n_layers > 0:
            self.init_e.requires_grad_(False)
            self.emb.requires_grad_(False)
            n = min(freeze_first_n_layers, len(self.update_es))
            for i in range(n):
                self.update_es[i].requires_grad_(False)
                self.update_vs[i].requires_grad_(False)
            self.condition_encoder.requires_grad_(False)
            self.nm_gamma_mlp.requires_grad_(False)
            self.nm_beta_mlp.requires_grad_(False)
            if self.atom_encoder is not None:
                self.atom_encoder.requires_grad_(False)
            if self.atom_gamma_mlp is not None:
                self.atom_gamma_mlp.requires_grad_(False)
                self.atom_beta_mlp.requires_grad_(False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_extra_node_feature:
            self.extra_emb.reset_parameters()
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        glorot_orthogonal(self.output_lin.weight, scale=2.0)
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()



    def _get_global_per_node(self, data, batch, device):
        if self.mof_global_encoder is None:
            return None
        if not (hasattr(data, "name") and data.name is not None):
            return None

        mof_embeddings = []
        names = data.name if isinstance(data.name, (list, tuple)) else [data.name]
        for name in names:
            name = str(name)
            mof_name = name.split("_")[0] if "_" in name else name
            mof_embeddings.append(self.mof_global_encoder(mof_name))
        while len(mof_embeddings) < len(data.natoms):
            mof_embeddings.append(mof_embeddings[-1])

        global_emb = torch.cat(mof_embeddings, dim=0).to(device=device)
        global_emb = torch.nan_to_num(global_emb, nan=0.0, posinf=0.0, neginf=0.0)
        return global_emb[batch]

    def _get_nm_per_node(self, data, batch, num_atoms, device):
        if hasattr(data, "condition") and data.condition is not None:
            cond = data.condition
            if isinstance(cond, torch.Tensor):
                cond = cond.to(device=device, dtype=torch.float32)
                if cond.dim() == 1:
                    cond = cond.view(1, -1)
            else:
                cond = torch.tensor(cond, device=device, dtype=torch.float32).view(1, -1)

            if cond.size(0) == 1 and cond.size(1) > self.base_nm_dim:
                n_cond = cond.size(1) // self.base_nm_dim
                if cond.size(1) % self.base_nm_dim == 0 and n_cond == len(data.natoms):
                    cond = cond.view(n_cond, self.base_nm_dim)

            if cond.size(0) == 1 and len(data.natoms) > 1:
                cond = cond.expand(len(data.natoms), -1)

            nm_encoded = self.condition_encoder(cond)
            return nm_encoded[batch]
        return torch.zeros(num_atoms, self.nm_encoded_dim, device=device, dtype=torch.float32)

    @conditional_grad(torch.enable_grad())
    def forward_embeddings(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(len(z), dtype=torch.long, device=pos.device)
        )
        num_atoms = z.size(0)

        if self.use_extra_node_feature and getattr(data, "node_feature", None) is not None:
            extra_node_feature = self.extra_emb(data.node_feature)
        else:
            extra_node_feature = None

        graph = self.generate_graph(data)
        edge_index = graph.edge_index
        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(
            pos, edge_index, num_atoms, use_torsion=True
        )

        emb = self.emb(dist, angle, torsion, idx_kj)

        e = self.init_e(z, extra_node_feature, emb, i, j)
        node_hidden = self.init_v(e, i, return_hidden=True)

        nm_per_node = self._get_nm_per_node(data, batch, num_atoms, pos.device)
        gamma_nm = self.nm_gamma_mlp(nm_per_node)
        beta_nm = self.nm_beta_mlp(nm_per_node)
        node_hidden = node_hidden * (1.0 + gamma_nm) + beta_nm

        if self.atom_encoder is not None and self.atom_gamma_mlp is not None:
            tags = data.tags if hasattr(data, "tags") else None
            atom_feat = self.atom_encoder(z, tags=tags).to(device=pos.device)
            gamma_a = self.atom_gamma_mlp(atom_feat)
            beta_a = self.atom_beta_mlp(atom_feat)
            node_hidden = node_hidden * (1.0 + gamma_a) + beta_a

        global_per_node = self._get_global_per_node(data, batch, pos.device)
        if global_per_node is not None and self.global_input_norm is not None:
            global_per_node = self.global_input_norm(global_per_node)

        graph_features = self.init_u(
            torch.zeros_like(scatter(node_hidden, batch, dim=0)),
            node_hidden,
            batch,
        )

        for layer_idx, (update_e, update_v, update_u) in enumerate(zip(self.update_es, self.update_vs, self.update_us)):
            e = update_e(e, emb, idx_kj, idx_ji)
            node_hidden = update_v(e, i, return_hidden=True)

            if (
                layer_idx == self.global_inject_layer
                and 0 <= self.global_inject_layer < len(self.update_es)
                and global_per_node is not None
                and self.global_gamma_mlp is not None
            ):
                gamma_g = self.global_gamma_mlp(global_per_node)
                beta_g = self.global_beta_mlp(global_per_node)
                node_hidden = node_hidden * (1.0 + gamma_g) + beta_g

            graph_features = update_u(graph_features, node_hidden, batch)

        if (
            self.global_inject_layer >= len(self.update_es)
            and global_per_node is not None
            and self.global_gamma_mlp is not None
        ):
            gamma_g = self.global_gamma_mlp(global_per_node)
            beta_g = self.global_beta_mlp(global_per_node)
            node_hidden = node_hidden * (1.0 + gamma_g) + beta_g

        return {
            "node_features": node_hidden,
            "graph_features": graph_features,
            "edge_features": e[0],
        }

    def _forward(self, data):
        emb = self.forward_embeddings(data)
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(len(data.atomic_numbers), dtype=torch.long, device=data.pos.device)
        )
        node_hidden = emb["node_features"]
        node_energy = self.output_lin(node_hidden)
        energy = torch.zeros(len(data.natoms), device=data.pos.device)
        energy.index_add_(0, batch, node_energy.view(-1))
        return energy

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)
        outputs = {"energy": energy}
        if self.regress_forces:
            forces = -1 * torch.autograd.grad(
                energy,
                data.pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
            outputs["forces"] = forces
        return outputs


@registry.register_model("spherenet_global_weighted_energy_head")
class SphereNetWeightedEnergyHead(nn.Module, HeadInterface):
    def __init__(self, backbone, reduce: str = "weighted_sum", weight_nn_hidden_dim: int = 64, **kwargs):
        super().__init__()
        if isinstance(backbone, dict):
            bb_cfg = dict(backbone)
            bb_name = bb_cfg.pop("name", "spherenet_global")
            backbone = registry.get_model_class(bb_name)(**bb_cfg)

        self.backbone = backbone
        self.reduce = reduce

        c = backbone.out_emb_channels
        self.energy_mlp = nn.Sequential(
            nn.Linear(c, weight_nn_hidden_dim),
            nn.SiLU(),
            nn.Linear(weight_nn_hidden_dim, weight_nn_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(weight_nn_hidden_dim // 2, 1),
        )
        self.weight_mlp = nn.Sequential(
            nn.Linear(c, weight_nn_hidden_dim),
            nn.SiLU(),
            nn.Linear(weight_nn_hidden_dim, 1),
            nn.Sigmoid(),
        ) if "weighted" in reduce else None

    def forward(self, data, emb: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        if emb is None:
            emb = self.backbone.forward_embeddings(data)

        h = emb["node_features"]
        node_energy = self.energy_mlp(h)

        if self.weight_mlp is not None:
            node_weights = self.weight_mlp(h)
            weighted_node_energy = node_energy * node_weights
        else:
            node_weights = torch.ones_like(node_energy)
            weighted_node_energy = node_energy

        energy = torch.zeros(len(data.natoms), device=data.pos.device)
        energy.index_add_(0, data.batch, weighted_node_energy.view(-1))

        if self.reduce in ("weighted_sum_normalized", "mean"):
            weight_sums = torch.zeros(len(data.natoms), device=data.pos.device)
            weight_sums.index_add_(0, data.batch, node_weights.view(-1))
            weight_sums = torch.clamp(weight_sums, min=1e-6)
            energy = energy / weight_sums

        return {"energy": energy}
