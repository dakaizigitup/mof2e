import torch
from fairchem.core.models.hierarchical_global_emb import MOFHierarchicalEncoder
import logging

logging.basicConfig(level=logging.INFO)

def test_encoder():
    # Use the real path to ensure we can load the excel file (assumes user environment has access)
    excel_path = "/home/dell/autodl-tmp/lorafair/data/MOF_embedding_all.xlsx"
    print(f"Loading encoder from {excel_path}...")
    
    try:
        # device='cpu' for testing
        encoder = MOFHierarchicalEncoder(excel_path=excel_path, device='cpu')
        print("Encoder initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize encoder: {e}")
        return

    # Test names, including some that might trigger multi-label logic if they exist in dataset
    # We rely on the internal dataframe being loaded.
    test_names = ["MOF_0", "MOF_1", "UNKNOWN"]
    # Ideally find a name that has multi-label metals if known, but random names will test basic path.
    # The actual dataframe content determines if multi-label logic is triggered, 
    # but the code should handle single label entries gracefully too (split by ',' results in 1 token).
    
    print("\nTesting forward pass...")
    try:
        embeddings = encoder(test_names)
        print(f"Output shape: {embeddings.shape}")
        
        if embeddings.shape == (3, 64):
            print("SUCCESS: Output dimension matches expected (3, 64).")
        else:
            print(f"FAILURE: Output dimension mismatch. Expected (3, 64), got {embeddings.shape}")
            
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_encoder()
