"""
Multi-Stream Fusion Model for MOF Adsorption Energy Prediction

This module implements a 4-stream multi-modal fusion architecture:
1. GNN_MOF: Processes MOF atomic coordinates (local geometry)
2. MoleculeLibrary: Processes adsorbate molecular structures (H2O/CO2 templates)
3. MLP_MOF: Processes MOF global embeddings (material properties)
4. CountEncoder: Processes adsorbate counts (statistical features)

The four streams are fused via cross-attention mechanisms.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from fairchem.core.models.escn import eSCN

logger = logging.getLogger(__name__)


# ============================================================================
# Encoder 1: GNN_MOF
# ============================================================================

class GNN_MOF(nn.Module):
    """
    Encoder for MOF atomic coordinates using eSCN.
    Supports loading pretrained weights.
    """
    def __init__(self, escn_config: dict, hidden_dim: int = 512, pretrained_path: Optional[str] = None):
        super().__init__()
        self.escn = eSCN(**escn_config)
        
        # Projection layer to match hidden_dim
        # eSCN outputs sphere_channels (128), we need hidden_dim (512)
        escn_output_dim = escn_config.get('sphere_channels', 128)
        self.projection = nn.Linear(escn_output_dim, hidden_dim)
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            logger.info(f"Loading pretrained eSCN weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load weights (strict=False to allow partial loading)
            missing_keys, unexpected_keys = self.escn.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys when loading pretrained eSCN: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading pretrained eSCN: {unexpected_keys[:5]}...")
            
            logger.info("Pretrained eSCN weights loaded successfully")
        
    def forward(self, data_mof):
        """
        Args:
            data_mof: PyG Batch object with MOF graphs
        
        Returns:
            h_mof: [N_total_atoms, hidden_dim] node features
        """
        output = self.escn.forward_embeddings(data_mof)
        
        # forward_embeddings returns a dict with 'sphere_values'
        # sphere_values shape: [N_atoms * num_sphere_samples, sphere_channels]
        # We need to pool back to [N_atoms, sphere_channels]
        sphere_values = output['sphere_values']  # [N_atoms * num_samples, channels]
        
        # Get number of atoms and sphere samples
        num_atoms = len(data_mof.atomic_numbers)
        num_samples = sphere_values.shape[0] // num_atoms
        
        # Reshape and pool: [N_atoms, num_samples, channels] -> [N_atoms, channels]
        h_mof = sphere_values.view(num_atoms, num_samples, -1).mean(dim=1)
        
        # Project to hidden_dim
        h_mof = self.projection(h_mof)  # [N_atoms, hidden_dim]
        
        return h_mof


# ============================================================================
# Encoder 2: MoleculeLibrary
# ============================================================================

class SimpleMoleculeEncoder(nn.Module):
    """
    Lightweight GNN for encoding small molecules (H2O, CO2).
    Uses a simple message passing scheme.
    """
    def __init__(self, hidden_dim=512, num_layers=2):
        super().__init__()
        self.atom_embedding = nn.Embedding(10, hidden_dim)  # Support H, C, O, etc.
        
        # Message passing layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ) for _ in range(num_layers)
        ])
        
    def forward(self, pos, atomic_numbers):
        """
        Args:
            pos: [batch, N_atoms, 3] atomic positions
            atomic_numbers: [batch, N_atoms] atomic numbers
        
        Returns:
            h: [batch, hidden_dim] pooled molecular features
        """
        batch_size, n_atoms, _ = pos.shape
        
        # Atom embeddings
        h = self.atom_embedding(atomic_numbers)  # [batch, N_atoms, hidden_dim]
        
        # Simple message passing (mean aggregation)
        for layer in self.layers:
            h_agg = h.mean(dim=1, keepdim=True).expand(-1, n_atoms, -1)
            h = h + layer(h_agg)
        
        # Global pooling
        h_pooled = h.mean(dim=1)  # [batch, hidden_dim]
        return h_pooled


class MoleculeLibrary(nn.Module):
    """
    Molecular structure encoder using predefined templates.
    Encodes H2O and CO2 standard geometries + counts.
    """


class MoleculeLibrary(nn.Module):
    """
    Library of molecular templates (H2O, CO2).
    """
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        # Molecular templates (standard geometries in Angstroms)
        self.register_buffer('h2o_template', torch.tensor([
            [0.0000,  0.0000,  0.1173],  # O
            [0.0000,  0.7572, -0.4692],  # H
            [0.0000, -0.7572, -0.4692],  # H
        ]))
        
        self.register_buffer('co2_template', torch.tensor([
            [0.0000,  0.0000,  0.0000],  # C
            [1.1600,  0.0000,  0.0000],  # O
            [-1.1600, 0.0000,  0.0000],  # O
        ]))
        
        self.register_buffer('h2o_atoms', torch.tensor([8, 1, 1]))  # O, H, H
        self.register_buffer('co2_atoms', torch.tensor([6, 8, 8]))  # C, O, O
        
        # Molecular encoders
        self.h2o_encoder = SimpleMoleculeEncoder(hidden_dim=hidden_dim, num_layers=2)
        self.co2_encoder = SimpleMoleculeEncoder(hidden_dim=hidden_dim, num_layers=2)
        
        # Reverted to nn.Embedding based on analysis that counts are low (0, 1, 2)
        # Increased capacity to 100 to avoid future truncation
        self.count_embedding = nn.Embedding(100, hidden_dim)
        
    def forward(self, n_h2o, n_co2):
        """
        Args:
            n_h2o: [batch] number of H2O molecules
            n_co2: [batch] number of CO2 molecules
        
        Returns:
            h_ads: [batch, 2, hidden_dim] stacked [H2O_features, CO2_features]
        """
        batch_size = n_h2o.shape[0]
        
        # Encode H2O geometry
        h2o_pos = self.h2o_template.unsqueeze(0).expand(batch_size, -1, -1)
        h2o_atoms = self.h2o_atoms.unsqueeze(0).expand(batch_size, -1)
        h_h2o_geom = self.h2o_encoder(h2o_pos, h2o_atoms)  # [batch, hidden_dim]
        
        # Encode CO2 geometry
        co2_pos = self.co2_template.unsqueeze(0).expand(batch_size, -1, -1)
        co2_atoms = self.co2_atoms.unsqueeze(0).expand(batch_size, -1)
        h_co2_geom = self.co2_encoder(co2_pos, co2_atoms)  # [batch, hidden_dim]
        
        # Add count information (With safety clamping)
        n_h2o = n_h2o.clamp(0, 99)
        n_co2 = n_co2.clamp(0, 99)
        
        h_h2o = h_h2o_geom + self.count_embedding(n_h2o)
        h_co2 = h_co2_geom + self.count_embedding(n_co2)
        
        # Stack: [batch, 2, hidden_dim]
        h_ads = torch.stack([h_h2o, h_co2], dim=1)
        
        return h_ads


# ============================================================================
# Encoder 3: MLP_MOF
# ============================================================================

class MLP_MOF(nn.Module):
    """
    Encoder for MOF global embeddings (material properties).
    """
    def __init__(self, input_dim=64, hidden_dim=256, output_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, mof_embedding):
        """
        Args:
            mof_embedding: [batch, input_dim] global MOF features
        
        Returns:
            h_mof_global: [batch, output_dim]
        """
        return self.mlp(mof_embedding)


# ============================================================================
# Encoder 4: CountEncoder
# ============================================================================

class CountEncoder(nn.Module):
    """
    Encoder for adsorbate counts (statistical features).
    """
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, count_vector):
        """
        Args:
            count_vector: [batch, 2] = [n_H2O, n_CO2]
        
        Returns:
            h_count: [batch, output_dim]
        """
        return self.mlp(count_vector)


# ============================================================================
# Encoder 5: AtomEmbeddingStream
# ============================================================================

class AtomEmbeddingStream(nn.Module):
    """
    Encoder for atom-level composition using physical/chemical properties.
    Uses Equiformer's AtomPropertyEncoder for scientifically-grounded features.
    """
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        # Import AtomPropertyEncoder from Equiformer
        from fairchem.core.models.equiformer_v2.atomic_emb import AtomPropertyEncoder
        
        # Use Equiformer's encoder (13 physical features: atomic_radius, electronegativity, etc.)
        self.atom_encoder = AtomPropertyEncoder(max_Z=100, out_dim=128)
        
        # Statistical pooling: mean + std + max + min (preserves more information)
        self.pooling = nn.Sequential(
            nn.Linear(128 * 4, 256),  # 4 statistics × 128 features
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, hidden_dim)
        )
        
    def forward(self, atomic_numbers, batch_idx, batch_size, tags=None):
        """
        Args:
            atomic_numbers: [N_total_atoms] atomic numbers (1-100)
            batch_idx: [N_total_atoms] batch assignment
            batch_size: int
            tags: [N_total_atoms] optional atom tags for PhAST embedding
        
        Returns:
            h_atom: [batch, 1, hidden_dim] atom composition features
        """
        # Get physical/chemical features for each atom
        # AtomPropertyEncoder uses 13 features: atomic_number, atomic_radius, 
        # atomic_volume, atomic_weight, block, group, period, series,
        # electron_affinity, electronegativity, nvalence, econf, symbol
        # Convert to long for indexing
        atom_feats = self.atom_encoder(atomic_numbers.long(), tags=tags)  # [N_total_atoms, 128]
        
        # Statistical pooling (preserves distribution information)
        h_atom_list = []
        for i in range(batch_size):
            mask = batch_idx == i
            if mask.sum() == 0:
                # Handle empty batch (shouldn't happen, but for safety)
                h_atom_list.append(torch.zeros(128 * 4, device=atom_feats.device))
                continue
                
            feats_i = atom_feats[mask]  # [N_i, 128]
            
            # Compute 4 statistical features
            mean_feat = feats_i.mean(dim=0)  # [128] - average composition
            std_feat = feats_i.std(dim=0)    # [128] - composition variance
            max_feat = feats_i.max(dim=0)[0] # [128] - heaviest/most electronegative
            min_feat = feats_i.min(dim=0)[0] # [128] - lightest/least electronegative
            
            # Concatenate all statistics
            h_atom_i = torch.cat([mean_feat, std_feat, max_feat, min_feat])  # [512]
            h_atom_list.append(h_atom_i)
        
        h_atom = torch.stack(h_atom_list)  # [batch, 512]
        
        # Project to hidden_dim
        h_atom_encoded = self.pooling(h_atom)  # [batch, hidden_dim]
        
        return h_atom_encoded.unsqueeze(1)  # [batch, 1, hidden_dim]


# ============================================================================
# Cross-Attention Module
# ============================================================================

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention with residual connection.
    """
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch, N_q, dim]
            key: [batch, N_k, dim]
            value: [batch, N_k, dim]
            mask: [batch, N_q, N_k] optional attention mask
        
        Returns:
            out: [batch, N_q, dim]
        """
        B, N_q, _ = query.shape
        N_k = key.shape[1]
        
        # Multi-head projection
        Q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention: [B, heads, N_q, N_k]
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Output: [B, N_q, dim]
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N_q, -1)
        out = self.out_proj(out)
        
        # Residual + Norm
        return self.norm(query + out)


# ============================================================================
# Fusion Module
# ============================================================================

class AttentionPooling(nn.Module):
    """
    Attention pooling layer for sequence data.
    Input: [batch, len, dim]
    Output: [batch, dim]
    Formula: sum(softmax(MLP(x)) * x)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1) # scalar score
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [batch, len, dim]
        if x.dim() != 3:
            return x # Already pooled or vector
        
        scores = self.attn_net(x) # [batch, len, 1]
        
        if mask is not None:
             # mask: true for valid
             scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
             
        weights = torch.softmax(scores, dim=1) # [batch, len, 1]
        pooled = (x * weights).sum(dim=1) # [batch, dim]
        return pooled

class FusionModule(nn.Module):
    """
    Improved Fusion Module:
    1. Attention Pooling for sequence streams
    2. Concat + Deep MLP for aggregation
    """
    def __init__(self, dim: int = 512, num_streams: int = 8):
        super().__init__()
        self.dim = dim
        self.num_streams = num_streams
        
        # Attention Pooling for sequence-based streams
        self.poolers = nn.ModuleList([AttentionPooling(dim) for _ in range(num_streams)])
        
        # Fusion MLP
        self.fusion_net = nn.Sequential(
            nn.Linear(dim * num_streams, dim * 4),
            nn.LayerNorm(dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(dim, dim // 2)
        )
        
    def forward(
        self,
        f12: torch.Tensor, f13: torch.Tensor, f34: torch.Tensor, f24: torch.Tensor,
        f21: torch.Tensor, f31: torch.Tensor, f43: torch.Tensor, f42: torch.Tensor,
        mof_mask: Optional[torch.Tensor] = None,
        f15: Optional[torch.Tensor] = None, f51: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        streams = []
        
        # Helper to process stream using index tracker
        idx = 0
        def process(tensor, mask=None):
            nonlocal idx
            out = tensor
            if tensor.dim() == 3:
                if tensor.shape[1] == 1:
                    out = tensor.squeeze(1)
                else:
                    curr_mask = None
                    if mask is not None and mask.shape[1] == tensor.shape[1]:
                        curr_mask = mask
                    out = self.poolers[idx](tensor, mask=curr_mask)
            
            idx += 1
            return out
            
        # 1. Forward streams (Order: 12, 13, 15, 34, 24)
        streams.append(process(f12, mof_mask)) # idx 0
        streams.append(process(f13, mof_mask)) # idx 1
        
        if self.num_streams == 10:
             if f15 is None: raise ValueError("Missing f15")
             streams.append(process(f15, mof_mask)) # idx 2
             
        streams.append(process(f34)) 
        streams.append(process(f24)) 
        
        # 2. Reverse streams (Order: 21, 31, 51, 43, 42)
        streams.append(process(f21))
        streams.append(process(f31))
        
        if self.num_streams == 10:
            if f51 is None: raise ValueError("Missing f51")
            streams.append(process(f51))
            
        streams.append(process(f43))
        streams.append(process(f42))
        
        # Concat
        f_concat = torch.cat(streams, dim=-1) # [batch, dim * num_streams]
        
        # Deep Fusion
        out = self.fusion_net(f_concat) # [batch, dim//2]
        
        return out


# ============================================================================
# Complete Multi-Stream Fusion Model
# ============================================================================

class MultiStreamFusionModel(nn.Module):
    """
    Complete 4-stream multi-modal fusion model.
    
    Args:
        escn_config: Configuration dict for eSCN
        hidden_dim: Hidden dimension for all encoders (default: 512)
        mof_emb_dim: Dimension of MOF global embedding (default: 64)
        pretrained_escn_path: Path to pretrained eSCN checkpoint (optional)
        freeze_escn: Whether to freeze eSCN weights (default: False)
        use_atom_embedding: Whether to use atom-level embedding stream (default: False)
    """
    def __init__(
        self,
        escn_config: dict,
        hidden_dim: int = 512,
        mof_emb_dim: int = 64,
        pretrained_escn_path: Optional[str] = None,
        freeze_escn: bool = False,
        use_atom_embedding: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_atom_embedding = use_atom_embedding
        
        # Five encoders
        self.gnn_mof = GNN_MOF(escn_config, hidden_dim=hidden_dim, pretrained_path=pretrained_escn_path)
        self.molecule_lib = MoleculeLibrary(hidden_dim=hidden_dim)
        self.mlp_mof = MLP_MOF(input_dim=mof_emb_dim, output_dim=hidden_dim)
        self.count_encoder = CountEncoder(input_dim=2, output_dim=hidden_dim)
        
        if use_atom_embedding:
            self.atom_embedding = AtomEmbeddingStream(hidden_dim=hidden_dim)
            logger.info("Atom Level Embedding: ENABLED")
        else:
            self.atom_embedding = None
            logger.info("Atom Level Embedding: DISABLED")
        
        # Freeze eSCN if requested
        if freeze_escn:
            logger.info("Freezing eSCN weights")
            for param in self.gnn_mof.parameters():
                param.requires_grad = False
        
        # Bidirectional cross-attention modules
        # Forward direction (original)
        self.ca12 = CrossAttention(dim=hidden_dim, num_heads=8)  # MOF → Ads
        self.ca13 = CrossAttention(dim=hidden_dim, num_heads=8)  # MOF → Global
        self.ca34 = CrossAttention(dim=hidden_dim, num_heads=8)  # Global → Count
        self.ca24 = CrossAttention(dim=hidden_dim, num_heads=8)  # Ads → Count
        
        # Reverse direction
        self.ca21 = CrossAttention(dim=hidden_dim, num_heads=8)  # Ads → MOF
        self.ca31 = CrossAttention(dim=hidden_dim, num_heads=8)  # Global → MOF
        self.ca43 = CrossAttention(dim=hidden_dim, num_heads=8)  # Count → Global
        self.ca42 = CrossAttention(dim=hidden_dim, num_heads=8)  # Count → Ads
        
        if use_atom_embedding:
            self.ca15 = CrossAttention(dim=hidden_dim, num_heads=8)  # MOF → Atom (NEW)
            self.ca51 = CrossAttention(dim=hidden_dim, num_heads=8)  # Atom → MOF (NEW)
            num_streams = 10
        else:
            self.ca15 = None
            self.ca51 = None
            num_streams = 8
        
        # Fusion module (now handles 8 or 10 streams)
        self.fusion = FusionModule(dim=hidden_dim, num_streams=num_streams)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, data_mof, mof_embedding, n_h2o, n_co2):
        """
        Args:
            data_mof: PyG Batch object (MOF graphs)
            mof_embedding: [batch, mof_emb_dim] global MOF features
            n_h2o: [batch] number of H2O molecules
            n_co2: [batch] number of CO2 molecules
        
        Returns:
            energy: [batch] predicted energies
        """
        batch_size = mof_embedding.shape[0]
        
        # 1. Five encoders
        h_mof = self.gnn_mof(data_mof)  # [N_total_atoms, hidden_dim]
        h_ads = self.molecule_lib(n_h2o, n_co2)  # [batch, 2, hidden_dim]
        h_mof_global = self.mlp_mof(mof_embedding).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Build count vector
        count_vector = torch.stack([n_h2o.float(), n_co2.float()], dim=-1)
        h_count = self.count_encoder(count_vector).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Atom embedding (Conditional)
        h_atom = None
        if self.use_atom_embedding:
            # AtomPropertyEncoder uses atomic_numbers directly (handling long conversion internally or via previous valid call)
            # Actually we fixed it to pass long.
            tags = data_mof.tags if hasattr(data_mof, "tags") else None
            h_atom = self.atom_embedding(data_mof.atomic_numbers, data_mof.batch, batch_size, tags=tags)
        
        # 2. Convert h_mof to batch format WITH MASK
        h_mof_batched, mof_mask = self._to_batch(h_mof, data_mof.batch, batch_size)
        
        # 3. Bidirectional cross-attentions (10 total)
        # Forward direction
        # CA12: MOF nodes attend to adsorbates
        f12 = self.ca12(query=h_mof_batched, key=h_ads, value=h_ads, mask=None)
        f12 = f12 * mof_mask.unsqueeze(-1)  # Apply mask
        
        # CA13: MOF nodes attend to global MOF features  
        f13 = self.ca13(query=h_mof_batched, key=h_mof_global, value=h_mof_global, mask=None)
        f13 = f13 * mof_mask.unsqueeze(-1)
        
        f15 = None
        if self.use_atom_embedding:
            # CA15: MOF nodes attend to atom composition (NEW)
            f15 = self.ca15(query=h_mof_batched, key=h_atom, value=h_atom, mask=None)
            f15 = f15 * mof_mask.unsqueeze(-1)
        
        # CA34: Global MOF attends to counts
        f34 = self.ca34(query=h_mof_global, key=h_count, value=h_count, mask=None)
        
        # CA24: Adsorbates attend to counts
        f24 = self.ca24(query=h_ads, key=h_count, value=h_count, mask=None)
        
        # Reverse direction
        # CA21: Adsorbates attend to MOF nodes
        f21 = self.ca21(query=h_ads, key=h_mof_batched, value=h_mof_batched, mask=None)
        
        # CA31: Global MOF attends to MOF nodes
        f31 = self.ca31(query=h_mof_global, key=h_mof_batched, value=h_mof_batched, mask=None)
        
        f51 = None
        if self.use_atom_embedding:
            # CA51: Atom composition attends to MOF nodes (NEW)
            f51 = self.ca51(query=h_atom, key=h_mof_batched, value=h_mof_batched, mask=None)
        
        # CA43: Counts attend to global MOF
        f43 = self.ca43(query=h_count, key=h_mof_global, value=h_mof_global, mask=None)
        
        # CA42: Counts attend to adsorbates
        f42 = self.ca42(query=h_count, key=h_ads, value=h_ads, mask=None)
        
        # 4. Fusion (8 or 10 streams)
        fused = self.fusion(
            f12, f13, f34, f24, f21, f31, f43, f42,
            mof_mask=mof_mask,
            f15=f15, f51=f51
        )  # [batch, hidden_dim//2]
        
        # 5. Prediction
        energy = self.output_head(fused)  # [batch, 1]
        return energy.squeeze(-1)
    
    def _to_batch(self, x, batch_idx, batch_size):
        """
        Convert node features to batch format [batch, max_nodes, dim] with attention mask.
        
        Args:
            x: [N_total_nodes, dim]
            batch_idx: [N_total_nodes] batch assignment
            batch_size: int
        
        Returns:
            batched: [batch_size, max_nodes, dim]
            mask: [batch_size, max_nodes] - True for real nodes, False for padding
        """
        max_nodes = (batch_idx.bincount()).max().item()
        dim = x.shape[-1]
        batched = torch.zeros(batch_size, max_nodes, dim, device=x.device, dtype=x.dtype)
        mask = torch.zeros(batch_size, max_nodes, device=x.device, dtype=torch.bool)
        
        for i in range(batch_size):
            batch_mask = batch_idx == i
            nodes = x[batch_mask]
            n_nodes = nodes.shape[0]
            batched[i, :n_nodes] = nodes
            mask[i, :n_nodes] = True  # Mark real nodes
        
        return batched, mask

