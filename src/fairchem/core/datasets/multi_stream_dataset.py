"""
Multi-Stream Dataset for 4-Stream Fusion Architecture

This dataset wraps the existing LMDB dataset and ExplodeYRelaxedDataset
to provide all inputs needed for the multi-stream fusion model:
1. MOF graph (coordinates, atomic numbers, edges)
2. MOF global embedding (from MOFGlobalPropertyEncoder)
3. Adsorbate molecule counts (n_H2O, n_CO2)
4. Condition vector [n_H2O, n_CO2]
5. Target energy
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Data

from fairchem.core.common.registry import registry
from fairchem.core.datasets.lmdb_dataset import LmdbDataset
from fairchem.core.datasets.base_dataset import BaseDataset, ExplodeYRelaxedDataset
# from fairchem.core.models.equiformer_v2.global_emb import MOFGlobalPropertyEncoder
from fairchem.core.models.hierarchical_global_emb import MOFHierarchicalEncoder

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@registry.register_dataset("multi_stream")
class MultiStreamDataset(BaseDataset):
    """
    Dataset for 4-stream multi-modal fusion architecture.
    
    Wraps ExplodeYRelaxedDataset and adds:
    - MOF global embeddings (from MOFGlobalPropertyEncoder)
    - Adsorbate molecule counts (n_H2O, n_CO2)
    
    Args:
        config (dict): Dataset configuration with keys:
            - src: Path to LMDB dataset
            - mof_embedding_path: Path to MOF_embedding_all.xlsx
            - global_dim: Dimension of global embedding (default: 64)
            - explode_y_relaxed: Config for ExplodeYRelaxedDataset (optional)
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # 1. Initialize base LMDB dataset
        logger.info(f"Loading LMDB dataset from {config['src']}")
        self.base_dataset = LmdbDataset({"src": config["src"]})
        
        # Use base_dataset directly - OCP will handle ExplodeYRelaxedDataset
        self.dataset = self.base_dataset
        
        # 3. Initialize MOF global encoder (Redesigned Hierarchical)
        mof_emb_path = config.get(
            "mof_embedding_path",
            "/home/dell/autodl-tmp/lorafair/data/MOF_embedding_all.xlsx"
        )
        # global_dim = config.get("global_dim", 64) # Hardcoded to 64 in HierarchicalEncoder for now
        
        logger.info(f"Loading MOF Hierarchical global encoder from {mof_emb_path}")
        self.mof_encoder = MOFHierarchicalEncoder(
            excel_path=mof_emb_path,
            device="cpu"  # Will move to GPU in model forward
        )
        
        self.mof_encoder.eval()
        
        # Set num_samples (required by BaseDataset)
        self.num_samples = len(self.dataset)
        
        logger.info(f"MultiStreamDataset initialized with {self.num_samples} samples")
    
    def __len__(self):
        length = len(self.dataset)
        logger.info(f"MultiStreamDataset.__len__() called, returning {length}")
        return length
    
    def __getitem__(self, idx):
        """
        Returns raw LMDB data with added MOF embedding.
        OCP's ExplodeYRelaxedDataset will handle the y_relaxed -> (condition, energy) conversion.
        
        Returns:
            Data object with:
                - All original LMDB attributes (pos, atomic_numbers, cell, y_relaxed, etc.)
                - mof_embedding: Tensor [64] (added by us)
                - mof_name: str (added for debugging)
        """
        # Get raw data from LMDB
        data = self.dataset[idx]
        
        # Extract MOF name
        if hasattr(data, 'name'):
            raw_name = data.name
            mof_name = raw_name.split('_')[0]
        elif hasattr(data, 'sid'):
            mof_name = data.sid.split('_')[0]
        else:
            logger.warning(f"Sample {idx} has no 'name' or 'sid' attribute, using 'UNKNOWN'")
            mof_name = "UNKNOWN"
        
        # Get global embedding
        with torch.no_grad():
            mof_embedding = self.mof_encoder([mof_name])  # [1, 64]
            mof_embedding = mof_embedding.squeeze(0)  # [64]
        
        # Add MOF embedding to data
        data.mof_embedding = mof_embedding
        data.mof_name = mof_name
        
        return data
    
    def close_db(self):
        """Close LMDB connections"""
        if hasattr(self.base_dataset, 'close_db'):
            self.base_dataset.close_db()
    
    def get_collate_fn(self):
        """Return the custom collate function for this dataset"""
        return collate_multi_stream


def collate_multi_stream(batch_list):
    """
    Custom collate function for MultiStreamDataset.
    
    Args:
        batch_list: List of dicts from __getitem__
    
    Returns:
        dict with batched tensors:
            - data_mof: PyG Batch object
            - mof_embedding: [batch, 64]
            - n_h2o: [batch]
            - n_co2: [batch]
            - condition: [batch, 2]
            - energy: [batch]
            - mof_names: List[str]
    """
    from torch_geometric.data import Batch
    
    # Batch MOF graphs
    data_mof_list = [item['data_mof'] for item in batch_list]
    data_mof_batch = Batch.from_data_list(data_mof_list)
    
    # Stack other tensors
    mof_embeddings = torch.stack([item['mof_embedding'] for item in batch_list])
    n_h2o = torch.tensor([item['n_h2o'] for item in batch_list], dtype=torch.long)
    n_co2 = torch.tensor([item['n_co2'] for item in batch_list], dtype=torch.long)
    conditions = torch.stack([item['condition'] for item in batch_list])
    energies = torch.tensor([item['energy'] for item in batch_list], dtype=torch.float32)
    mof_names = [item['mof_name'] for item in batch_list]
    
    return {
        'data_mof': data_mof_batch,
        'mof_embedding': mof_embeddings,
        'n_h2o': n_h2o,
        'n_co2': n_co2,
        'condition': conditions,
        'energy': energies,
        'mof_names': mof_names,
    }


# Example usage
if __name__ == "__main__":
    # Test the dataset
    config = {
        "src": "/home/dell/autodl-tmp/lorafair/data/mof2e/train",
        "mof_embedding_path": "/home/dell/autodl-tmp/lorafair/data/MOF_embedding_all.xlsx",
        "global_dim": 64,
    }
    
    dataset = MultiStreamDataset(config)
    print(f"Dataset length: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print("\nSample 0:")
    print(f"  MOF name: {sample['mof_name']}")
    print(f"  MOF atoms: {sample['data_mof'].pos.shape[0]}")
    print(f"  MOF embedding shape: {sample['mof_embedding'].shape}")
    print(f"  n_H2O: {sample['n_h2o']}, n_CO2: {sample['n_co2']}")
    print(f"  Condition: {sample['condition']}")
    print(f"  Energy: {sample['energy']:.4f}")
    
    # Test batching
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_multi_stream)
    batch = next(iter(loader))
    
    print("\nBatch:")
    print(f"  data_mof batch size: {batch['data_mof'].num_graphs}")
    print(f"  mof_embedding shape: {batch['mof_embedding'].shape}")
    print(f"  n_h2o shape: {batch['n_h2o'].shape}")
    print(f"  energy shape: {batch['energy'].shape}")
