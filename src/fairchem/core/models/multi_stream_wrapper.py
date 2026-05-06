"""
Multi-Stream Fusion Model Wrapper for OCP Trainer

This module provides a wrapper to integrate MultiStreamFusionModel
with the OCP training framework.
"""

import logging
import torch
import torch.nn as nn

from fairchem.core.common.registry import registry
from fairchem.core.models.base import GraphModelMixin
from fairchem.core.models.modular_multi_stream_model import ModularMOFModel
from fairchem.core.datasets.multi_stream_dataset import MultiStreamDataset, collate_multi_stream

logger = logging.getLogger(__name__)


@registry.register_model("multi_stream_fusion")
class MultiStreamFusionWrapper(nn.Module, GraphModelMixin):
    """
    Wrapper for MultiStreamFusionModel to work with OCP trainer.
    
    This wrapper handles:
    - Model initialization from config
    - Forward pass with OCP data format
    - Loss computation
    - Metric reporting
    """
    
    def __init__(
        self,
        use_pbc=True,
        regress_forces=False,
        otf_graph=True,
        hidden_dim=512,
        mof_emb_dim=64,
        pretrained_escn_path=None,
        freeze_escn=False,
        escn_config=None,
        # Modular Expert Configuration
        use_mole=True,
        use_feature_gating=False,
        use_flexibility=False,
        modular_config=None, # Clean way to pass dict
        **kwargs
    ):
        super().__init__()
        
        # Store OCP-required attributes
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        
        # Build eSCN config
        if escn_config is None:
            escn_config = {}
        
        # Ensure required parameters
        escn_config.setdefault('use_pbc', use_pbc)
        escn_config.setdefault('regress_forces', regress_forces)
        escn_config.setdefault('otf_graph', otf_graph)
        escn_config.setdefault('max_neighbors', 20)
        escn_config.setdefault('cutoff', 8.0)
        escn_config.setdefault('max_num_elements', 100)
        escn_config.setdefault('num_layers', 8)
        escn_config.setdefault('lmax_list', [6])
        escn_config.setdefault('mmax_list', [2])
        escn_config.setdefault('sphere_channels', 128)
        escn_config.setdefault('hidden_channels', 256)
        escn_config.setdefault('edge_channels', 128)
        escn_config.setdefault('num_sphere_samples', 128)
        escn_config.setdefault('distance_function', 'gaussian')
        escn_config.setdefault('basis_width_scalar', 2.0)
        escn_config.setdefault('distance_resolution', 0.02)
        
        
        if modular_config:
            # Override individual flags if config dict is provided
            use_mole = modular_config.get('use_mole', use_mole)
            use_feature_gating = modular_config.get('use_feature_gating', use_feature_gating)
            use_flexibility = modular_config.get('use_flexibility', use_flexibility)

        logger.info(f"Initializing ModularMOFModel (Expert Architecture) with hidden_dim={hidden_dim}")
        logger.info(f"Modular Config: MOLE={use_mole}, Gating={use_feature_gating}, Flex={use_flexibility}")

                
        # Initialize multi-stream model
        # Initialize modular expert model
        self.model = ModularMOFModel(
            escn_config=escn_config,
            hidden_dim=hidden_dim,
            mof_emb_dim=mof_emb_dim,
            pretrained_escn_path=pretrained_escn_path,
            freeze_escn=freeze_escn,
            use_atom_embedding=True, # Expert model requires this for Chemistry Expert
            # New Configs
            use_mole=use_mole,
            use_feature_gating=use_feature_gating,
            use_flexibility=use_flexibility
        )
        
        self.num_targets = 1  # Energy prediction
        
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, data):
        """
        Forward pass compatible with OCP trainer.
        
        Args:
            data: Batch from MultiStreamDataset with attributes:
                - pos, atomic_numbers, cell, etc. (MOF graph, batched by PyG)
                - mof_embedding: [batch*64] (batched, needs reshape)
                - condition: [batch*2] (batched, needs reshape)
                - energy: [batch] (target)
        
        Returns:
            dict with 'energy' key
        """
        # Get batch size from the batch attribute
        batch_size = data.batch.max().item() + 1 if hasattr(data, 'batch') else 1
        
        # Reshape mof_embedding from [batch*64] to [batch, 64]
        if hasattr(data, 'mof_embedding'):
            mof_embedding = data.mof_embedding.view(batch_size, -1)
        else:
            # Fallback: create dummy embedding
            mof_embedding = torch.zeros(batch_size, 64, device=data.pos.device)
            logger.warning("No mof_embedding found in data, using zeros")
        
        # Reshape condition from [batch*2] to [batch, 2] and extract counts
        if hasattr(data, 'condition'):
            condition = data.condition.view(batch_size, 2)
            n_h2o = condition[:, 0].long()
            n_co2 = condition[:, 1].long()
        else:
            # Fallback: use zeros
            n_h2o = torch.zeros(batch_size, dtype=torch.long, device=data.pos.device)
            n_co2 = torch.zeros(batch_size, dtype=torch.long, device=data.pos.device)
            logger.warning("No condition found in data, using zeros")
        
        # Forward pass
        energy = self.model(
            data_mof=data,
            mof_embedding=mof_embedding,
            n_h2o=n_h2o,
            n_co2=n_co2
        )
        
        return {
            'energy': energy
        }
    
    @torch.no_grad()
    def predict(self, data):
        """Prediction mode (same as forward for this model)"""
        return self.forward(data)



