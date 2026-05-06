
from __future__ import annotations

import logging
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

# Import existing encoders and building blocks
from fairchem.core.models.multi_stream_model import (
    GNN_MOF, 
    MoleculeLibrary, 
    MLP_MOF, 
    CountEncoder, 
    AtomEmbeddingStream, 
    CrossAttention,
    AttentionPooling
)

logger = logging.getLogger(__name__)

# ============================================================================
# MOLE Layer (Mixture of Linear Experts)
# ============================================================================
class MoleLayer(nn.Module):
    """
    Mixture of Linear Experts Layer.
    Gate decides which linear expert to use for each input.
    Output = Sum(Gate_i(x) * Expert_i(x))
    """
    def __init__(self, in_features, out_features, num_experts=4, use_bias=True):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        
        # Experts: Tensor of shape (num_experts, in_features, out_features)
        self.expert_weights = nn.Parameter(torch.empty(num_experts, in_features, out_features))
        if use_bias:
            self.expert_biases = nn.Parameter(torch.empty(num_experts, out_features))
        else:
            self.register_parameter('expert_biases', None)
            
        # Gating Network
        self.gate = nn.Linear(in_features, num_experts)
        self.softmax = nn.Softmax(dim=-1)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.expert_weights)
        if self.expert_biases is not None:
            nn.init.constant_(self.expert_biases, 0.0)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 0.0)

    def forward(self, x):
        # x: [batch_size, in_features]
        gates = self.softmax(self.gate(x)) # [B, num_experts]
        
        # expert_out: [B, num_experts, out_features]
        expert_out = torch.einsum('bi,eio->beo', x, self.expert_weights)
        if self.expert_biases is not None:
             expert_out = expert_out + self.expert_biases # Broadcasting
        
        # Weighted Sum
        output = torch.einsum('be,beo->bo', gates, expert_out)
        return output


# ============================================================================
# Expert 1: Geometry Expert (The "Architect")
# ============================================================================
class GeometryExpert(nn.Module):
    """
    Expert dealing with Spatial/Geometric constraints (Steric hindrance).
    Core interaction: MOF Structure (GNN) <-> Adsorbate Shape (MolLib)
    """
    def __init__(self, dim: int = 512, num_heads: int = 8, use_mole: bool = True):
        super().__init__()
        # Bidirectional attention between MOF nodes and Adsorbates
        self.mof_to_ads = CrossAttention(dim=dim, num_heads=num_heads)
        self.ads_to_mof = CrossAttention(dim=dim, num_heads=num_heads)
        
        # Expert-specific pooling and projection
        self.poolout = AttentionPooling(dim)
        self.project = nn.Sequential(
            nn.Linear(dim * 2, dim), # Concat(MOF_view, Ads_view)
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.GELU(),
        )
        if use_mole:
            self.project.add_module('mole', MoleLayer(dim, dim, num_experts=4))
        else:
             self.project.add_module('linear_out', nn.Linear(dim, dim))

        
    def forward(self, h_mof: torch.Tensor, h_ads: torch.Tensor, mof_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_mof: [B, N_mof, D] (Batched Node Features)
            h_ads: [B, N_ads, D] (Adsorbate Features)
            mof_mask: [B, N_mof]
        Returns:
            z_geo: [B, D] - The "Steric Vector"
        """
        # 1. MOF attends to Adsorbates ("Is there space for these?")
        # h_ads is usually [B, 2, D] (H2O, CO2)
        v_mof = self.mof_to_ads(query=h_mof, key=h_ads, value=h_ads) # [B, N_mof, D]
        v_mof = v_mof * mof_mask.unsqueeze(-1)
        
        # 2. Adsorbates attend to MOF ("Can I fit in here?")
        v_ads = self.ads_to_mof(query=h_ads, key=h_mof, value=h_mof, mask=None) # [B, N_ads, D]
        
        # 3. Pooling
        # Pool MOF view (reduce N_mof -> 1)
        # Use simple masked mean for stability or AttentionPooling? 
        # Let's use AttentionPooling for specific features
        z_mof = self.poolout(v_mof, mask=mof_mask) # [B, D]
        
        # Pool Ads view (reduce N_ads -> 1)
        if v_ads.size(1) > 1:
             z_ads = v_ads.mean(dim=1) # [B, D] 
        else:
             z_ads = v_ads.squeeze(1)
             
        # 4. Fusion to Expert Vector
        z_geo = self.project(torch.cat([z_mof, z_ads], dim=-1))
        return z_geo

# ============================================================================
# Expert 2: Chemistry Expert (The "Chemist")
# ============================================================================
class ChemistryExpert(nn.Module):
    """
    Expert dealing with Chemical Affinity.
    Core interaction: Atom Properties (AtomEmb) <-> Adsorbate Chemical Props
    Also considers Global MOF chemical props (from MLP_MOF)
    """
    def __init__(self, dim: int = 512, num_heads: int = 8, use_mole: bool = True):
        super().__init__()
        # Interaction: Atom <-> Adsorbate
        # Weighted Reasoning: Learn which atom-adsorbate pairs matter most
        
        # 1. Feature Interaction Layer
        # Instead of standard Attention, we use a Gated Interaction Network
        self.atom_proj = nn.Linear(dim, dim)
        self.ads_proj = nn.Linear(dim, dim)
        
        # 2. Importance Weight Network (The "Judge")
        # Determines the importance of each interaction pair
        self.weight_net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Bound [0, 1] importance score
        )
        
        # 3. Reasoning Layer (Deep Feature Transformation)
        self.reasoning = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        # MOLE Projection for Final Decision
        if use_mole:
            self.project = MoleLayer(dim, dim, num_experts=4)
        else:
            self.project = nn.Linear(dim, dim)

        
    def forward(self, h_atom: torch.Tensor, h_ads: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_atom: [B, 1, D] (Backbone pooled atom feat) OR [B, N, D] if sequence
            h_ads: [B, N_ads, D] (Adsorbate features)
        Returns:
            z_chem: [B, D]
        """
        # 1. Expand features for pairwise interaction
        # h_atom: [B, 1, D] -> [B, N_ads, D] (Broadcast)
        # h_ads:  [B, N_ads, D]
        batch_size, n_ads, dim = h_ads.shape
        
        h_atom_expand = h_atom.expand(-1, n_ads, -1)
        
        # 2. Interactive features (Element-wise fusion)
        # "Chemistry happens when orbitals overlap" -> Hadamard product + Projection
        feat_inter = self.atom_proj(h_atom_expand) * self.ads_proj(h_ads) # [B, N_ads, D]
        
        # 3. Compute Importance Weights ("Which adsorbate/site is active?")
        weights = self.weight_net(feat_inter) # [B, N_ads, 1]
        
        # 4. Weighted Reasoning (The PhAST Logic: Weighted Sum of Features)
        # Focus heavily on the important pairs, ignore the noise
        z_weighted = (feat_inter * weights).sum(dim=1) # [B, D] Sum over adsorbates
        
        # Normalize by sum of weights to keep scale consistent
        weight_sum = weights.sum(dim=1).clamp(min=1e-6)
        z_weighted = z_weighted / weight_sum
        
        # 5. Deep Reasoning & Projection
        z_out = self.reasoning(z_weighted)
        
        return self.project(z_out)

# ============================================================================
# Expert 3: Interaction Expert (The "Statistician")
# ============================================================================
class InteractionExpert(nn.Module):
    """
    Expert dealing with Capacity/Load statistics.
    Core interaction: Count <-> Global Props
    """
    def __init__(self, dim: int = 512, num_heads: int = 8, use_mole: bool = True):
        super().__init__()
        
        self.global_to_count = CrossAttention(dim=dim, num_heads=num_heads)
        self.count_to_global = CrossAttention(dim=dim, num_heads=num_heads)
        
        self.bilinear = nn.Bilinear(dim, dim, dim)
        self.reduce = nn.Linear( dim * 2, dim ) # fallback
        
        # MOLE Projection
        if use_mole:
            self.project = MoleLayer(dim, dim, num_experts=4)
        else:
             self.project = nn.Linear(dim, dim)

        
    def forward(self, h_global: torch.Tensor, h_count: torch.Tensor) -> torch.Tensor:
        """
        Args:
             h_global: [B, 1, D]
             h_count: [B, 1, D]
        """
        # "Given this surface area (Global), can we fit this many (Count)?"
        v_gc = self.global_to_count(query=h_global, key=h_count, value=h_count)
        v_cg = self.count_to_global(query=h_count, key=h_global, value=h_global)
        
        v_gc = v_gc.squeeze(1)
        v_cg = v_cg.squeeze(1)
        
        # Bilinear Interaction for capacity modeling
        z_inter = self.bilinear(v_gc, v_cg)
        
        # Apply MOLE projection
        return self.project(z_inter)
# ============================================================================
# Expert 4: Flexibility Expert (The "Dynamicist") - Phase 3
# ============================================================================
class FlexibilityExpert(nn.Module):
    """
    Expert dealing with structural flexibility and long-range interactions.
    """
    def __init__(self, dim: int = 512, num_heads: int = 8, use_mole: bool = True):
        super().__init__()
        # Cross Attention: MOF <-> Ads
        self.mof_to_ads = CrossAttention(dim=dim, num_heads=num_heads)
        
        # Self Attention on MOF nodes to capture internal deformation potential
        self.mof_self_attn = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*2, batch_first=True)
        self.mof_encoder = nn.TransformerEncoder(self.mof_self_attn, num_layers=2)
        
        self.poolout = AttentionPooling(dim)
        
        # Expert Projection
        self.project = nn.Sequential(
             nn.Linear(dim, dim),
             nn.LayerNorm(dim),
             nn.GELU()
        )
        if use_mole:
            self.project.add_module('mole', MoleLayer(dim, dim, num_experts=4))
        else:
             self.project.add_module('linear_out', nn.Linear(dim, dim))
             
    def forward(self, h_mof: torch.Tensor, h_ads: torch.Tensor, mof_mask: torch.Tensor) -> torch.Tensor:
        # 1. MOF attends to adsorbates
        v_mof = self.mof_to_ads(query=h_mof, key=h_ads, value=h_ads) # [B, N, D]
        v_mof = v_mof * mof_mask.unsqueeze(-1)
        
        # 2. MOF Self-Attention to assess flexibility/rigidity
        v_flex = self.mof_encoder(v_mof, src_key_padding_mask=~mof_mask)
        
        # 3. Pooling
        z_flex = self.poolout(v_flex, mask=mof_mask)
        
        return self.project(z_flex)


# ============================================================================
# Director Gate


# ============================================================================
# Director Gate
# ============================================================================
class DirectorGate(nn.Module):
    """
    Context-Aware Gating mechanism.
    Decides trust weights for each expert.
    """
    def __init__(self, dim: int = 512, num_experts: int = 3, use_feature_gating: bool = False):
        super().__init__()
        self.use_feature_gating = use_feature_gating
        
        input_dim = dim * num_experts
        if use_feature_gating:
             # Add global features dimension. Assuming global features are [B, 1, D] -> [B, D]
             input_dim += dim 
             
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, experts: List[torch.Tensor], global_feat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # experts: list of [B, D]
        stacked = torch.cat(experts, dim=-1) # [B, D * num_experts]
        
        if self.use_feature_gating:
            if global_feat is None:
                raise ValueError("global_feat is required when use_feature_gating=True")
                
            # global_feat usually [B, 1, D], squeeze to [B, D]
            g_vec = global_feat.squeeze(1) if global_feat.dim() == 3 else global_feat
            
            stacked = torch.cat([stacked, g_vec], dim=-1)
            
        weights = self.gate_net(stacked) # [B, num_experts]
        
        # Weighted sum for final output
        # weights[:, 0] * experts[0] ...
        out = torch.zeros_like(experts[0])
        for i, expert_out in enumerate(experts):
            out += weights[:, i:i+1] * expert_out
            
        return out, weights

# ============================================================================
# Main Model
# ============================================================================
class ModularMOFModel(nn.Module):
    def __init__(
        self,
        escn_config: dict,
        hidden_dim: int = 512,
        mof_emb_dim: int = 64,
        pretrained_escn_path: Optional[str] = None,
        freeze_escn: bool = False,
        use_atom_embedding: bool = True, # Force True for Chem Expert
        # Configurable Flags
        use_mole: bool = True,
        use_feature_gating: bool = False,
        use_flexibility: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_mole = use_mole
        self.use_feature_gating = use_feature_gating
        self.use_flexibility = use_flexibility
        
        # --- 1. Encoders Layer (The Intelligence Gathering) ---
        self.gnn_mof = GNN_MOF(escn_config, hidden_dim=hidden_dim, pretrained_path=pretrained_escn_path)
        self.molecule_lib = MoleculeLibrary(hidden_dim=hidden_dim)
        self.mlp_mof = MLP_MOF(input_dim=mof_emb_dim, output_dim=hidden_dim)
        self.count_encoder = CountEncoder(input_dim=2, output_dim=hidden_dim)
        
        if use_atom_embedding:
            self.atom_embedding = AtomEmbeddingStream(hidden_dim=hidden_dim)
        else:
            raise ValueError("ModularMOFModel requires atom_embedding=True for Chemistry Expert")
            
        if freeze_escn:
            for param in self.gnn_mof.parameters():
                param.requires_grad = False
                
        # --- 2. Experts Layer (The Specialized Reasoning) ---
        # --- 2. Experts Layer (The Specialized Reasoning) ---
        self.geo_expert = GeometryExpert(dim=hidden_dim, use_mole=use_mole)
        self.chem_expert = ChemistryExpert(dim=hidden_dim, use_mole=use_mole)
        self.inter_expert = InteractionExpert(dim=hidden_dim, use_mole=use_mole)
        
        if use_flexibility:
             self.flex_expert = FlexibilityExpert(dim=hidden_dim, use_mole=use_mole)
             num_experts = 4
        else:
             self.flex_expert = None
             num_experts = 3
        
        # --- 3. Management Layer (The Decision) ---
        self.director = DirectorGate(dim=hidden_dim, num_experts=num_experts, use_feature_gating=use_feature_gating)
        
        # Output Head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
    def _to_batch(self, x, batch_idx, batch_size):
        # Helper copied from multi_stream_model to batch GNN output
        max_nodes = (batch_idx.bincount()).max().item()
        dim = x.shape[-1]
        batched = torch.zeros(batch_size, max_nodes, dim, device=x.device, dtype=x.dtype)
        mask = torch.zeros(batch_size, max_nodes, device=x.device, dtype=torch.bool)
        
        for i in range(batch_size):
            batch_mask = batch_idx == i
            nodes = x[batch_mask]
            n_nodes = nodes.shape[0]
            batched[i, :n_nodes] = nodes
            mask[i, :n_nodes] = True 
        return batched, mask
        
    def forward(self, data_mof, mof_embedding, n_h2o, n_co2):
        batch_size = mof_embedding.shape[0]
        
        # --- Phase 1: Encoding ---
        # 1. Geometry features
        h_mof_nodes = self.gnn_mof(data_mof) # [N_total, D]
        h_mof_batch, mof_mask = self._to_batch(h_mof_nodes, data_mof.batch, batch_size) # [B, N_max, D]
        
        # 2. Molecule features
        h_ads = self.molecule_lib(n_h2o, n_co2) # [B, 2, D]
        
        # 3. Chemical features (Atom Level)
        # AtomEmbeddingStream expects valid atomic numbers.
        tags = data_mof.tags if hasattr(data_mof, "tags") else None
        h_atom = self.atom_embedding(data_mof.atomic_numbers, data_mof.batch, batch_size, tags=tags) # [B, 1, D]
        
        # 4. Global/Count features
        h_global = self.mlp_mof(mof_embedding).unsqueeze(1) # [B, 1, D]
        count_vec = torch.stack([n_h2o.float(), n_co2.float()], dim=-1)
        h_count = self.count_encoder(count_vec).unsqueeze(1) # [B, 1, D]
        
        # --- Phase 2: Expert Reasoning ---
        # 🏛️ Geometry Expert: MOF Nodes + Adsorbate Shapes
        z_geo = self.geo_expert(h_mof_batch, h_ads, mof_mask) # [B, D]
        
        # 🧪 Chemistry Expert: Atom Properties + Adsorbate Chem
        z_chem = self.chem_expert(h_atom, h_ads) # [B, D]
        
        # 📊 Interaction Expert: Global Props + Counts
        z_inter = self.inter_expert(h_global, h_count) # [B, D]
        
        # --- Phase 3: Director Decision ---
        expert_outputs = [z_geo, z_chem, z_inter]
        if self.use_flexibility:
             z_flex = self.flex_expert(h_mof_batch, h_ads, mof_mask)
             expert_outputs.append(z_flex)
             
        # Pass global features (h_global) to director if gating is enabled
        z_final, weights = self.director(expert_outputs, global_feat=h_global if self.use_feature_gating else None)
        
        # Final Prediction
        energy = self.output_head(z_final)
        return energy.squeeze(-1)
