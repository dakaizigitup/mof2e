#!/usr/bin/env python
"""
Debug script to test MultiStreamDataset and DataLoader
"""

import torch
from torch.utils.data import DataLoader
from fairchem.core.datasets.multi_stream_dataset import MultiStreamDataset
from fairchem.core.datasets import data_list_collater

# Test dataset
config = {
    "src": "/home/dell/autodl-tmp/lorafair/data/mof2e/train",
    "mof_embedding_path": "/home/dell/autodl-tmp/lorafair/data/MOF_embedding_all.xlsx",
    "global_dim": 64,
}

print("Creating dataset...")
dataset = MultiStreamDataset(config)
print(f"Dataset length: {len(dataset)}")
print(f"Dataset num_samples: {dataset.num_samples}")

# Test single sample
print("\nTesting single sample...")
sample = dataset[0]
print(f"Sample type: {type(sample)}")
print(f"Sample attributes: {sample.keys if hasattr(sample, 'keys') else dir(sample)}")
if hasattr(sample, 'mof_embedding'):
    print(f"  mof_embedding shape: {sample.mof_embedding.shape}")
if hasattr(sample, 'n_h2o'):
    print(f"  n_h2o: {sample.n_h2o}")
if hasattr(sample, 'n_co2'):
    print(f"  n_co2: {sample.n_co2}")

# Test DataLoader with default collate
print("\nTesting DataLoader with data_list_collater...")
try:
    loader = DataLoader(
        dataset, 
        batch_size=2, 
        collate_fn=data_list_collater,
        num_workers=0
    )
    print(f"DataLoader length: {len(loader)}")
    
    # Try to get first batch
    print("\nGetting first batch...")
    batch = next(iter(loader))
    print(f"Batch type: {type(batch)}")
    print(f"Batch num_graphs: {batch.num_graphs if hasattr(batch, 'num_graphs') else 'N/A'}")
    if hasattr(batch, 'mof_embedding'):
        print(f"Batch mof_embedding shape: {batch.mof_embedding.shape}")
    print("SUCCESS!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
