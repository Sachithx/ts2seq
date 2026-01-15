"""
PIX2SEQ VIT ENCODER - PYTORCH IMPLEMENTATION
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from typing import Optional, Dict

# ============================================================================
# MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention matching Pix2Seq structure."""
    
    def __init__(self, hidden_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = 64  # From your weights:  (12, 64)
        
        # Q, K, V projections
        # TF shape: (768, 12, 64), PyTorch:  needs (768, 12*64=768)
        self.query_dense = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=True)
        self.key_dense = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=True)
        self.value_dense = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=True)
        
        # Output projection
        # TF shape: (12, 64, 768), PyTorch: (12*64=768, 768)
        self.output_dense = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x:  [batch, seq_len, hidden_dim]
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.query_dense(x)  # [B, L, 768]
        K = self.key_dense(x)
        V = self.value_dense(x)
        
        # Reshape for multi-head attention
        # [B, L, 768] -> [B, L, 12, 64] -> [B, 12, L, 64]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, 12, L, 64]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, L, 12, 64]
        attn_output = attn_output.view(batch_size, seq_len, -1)  # [B, L, 768]
        
        # Output projection
        output = self.output_dense(attn_output)
        
        return output

# ============================================================================
# ATTENTION POOLING
# ============================================================================

class AttentionPooling(nn.Module):
    """
    Attention-based pooling to aggregate patch features.
    Learns which patches are important for classification.
    """
    
    def __init__(self, hidden_dim=768, num_heads=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Attention scoring network
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, num_heads)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, num_patches, hidden_dim]
        
        Returns:
            pooled: [batch, hidden_dim]
        """
        # Compute attention scores for each patch
        # [B, num_patches, hidden_dim] -> [B, num_patches, num_heads]
        attn_scores = self.attention(x)
        
        # Softmax over patches dimension
        # [B, num_patches, num_heads]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        if self.num_heads == 1:
            # Single attention head
            # [B, num_patches, 1] * [B, num_patches, hidden_dim] -> [B, hidden_dim]
            pooled = (x * attn_weights).sum(dim=1)
        else:
            # Multi-head attention pooling
            # [B, num_patches, num_heads, 1] * [B, num_patches, 1, hidden_dim]
            attn_weights = attn_weights.unsqueeze(-1)  # [B, num_patches, num_heads, 1]
            x_expanded = x.unsqueeze(2)  # [B, num_patches, 1, hidden_dim]
            
            # Weighted sum per head
            # [B, num_patches, num_heads, hidden_dim] -> [B, num_heads, hidden_dim]
            pooled = (attn_weights * x_expanded).sum(dim=1)
            
            # Average across heads
            # [B, num_heads, hidden_dim] -> [B, hidden_dim]
            pooled = pooled.mean(dim=1)
        
        return pooled


# ============================================================================
# MLP (FEED-FORWARD)
# ============================================================================

class MLP(nn.Module):
    """MLP block with layer norm."""
    
    def __init__(self, hidden_dim=768, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_dim, mlp_dim)
        self.dense2 = nn.Linear(mlp_dim, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, L, 768]
        residual = x
        
        x = self.dense1(x)
        x = F.gelu(x)  # GELU activation
        x = self.dropout(x)
        
        x = self.dense2(x)
        x = self.dropout(x)
        
        # Add residual and normalize
        x = self.layernorm(x + residual)
        
        return x


# ============================================================================
# TRANSFORMER ENCODER LAYER
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(self, hidden_dim=768, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        
        # Multi-head attention with pre-layer norm
        self.mha_ln = nn.LayerNorm(hidden_dim)
        self.mha = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        # MLP with built-in post-layer norm
        self.mlp = MLP(hidden_dim, mlp_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x:  [batch, seq_len, hidden_dim]
        Returns: 
            x: [batch, seq_len, hidden_dim]
        """
        # MHA with pre-norm and residual
        residual = x
        x = self.mha_ln(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + residual
        
        # MLP (has its own norm and residual)
        x = self.mlp(x)
        
        return x


# ============================================================================
# COMPLETE VIT ENCODER
# ============================================================================

class Pix2SeqViTEncoder(nn.Module):
    """
    Vision Transformer encoder from Pix2Seq checkpoint. 
    
    Architecture:
        - Stem: 8x8 conv (stride 8) + LayerNorm
        - 12 Transformer encoder layers
        - Output LayerNorm
    """
    
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 hidden_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_dim=3072,
                 dropout=0.1,
                 pretrained_weights_path:  Optional[str] = None):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Stem:  Convolutional patch embedding
        self.stem_conv = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )
        self.stem_ln = nn.LayerNorm(hidden_dim)
        
        # Positional embedding (learnable)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_dim)
        )
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer norm
        self.output_ln = nn.LayerNorm(hidden_dim)
        
        # Initialize
        self._init_weights()
        
        # Load pretrained if provided
        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path)
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Initialize other layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x:  [batch, 3, H, W] input images
        Returns:
            features: [batch, num_patches, hidden_dim] or [batch, hidden_dim] if pooled
        """
        batch_size = x.shape[0]
        
        # Patch embedding:  [B, 3, H, W] -> [B, 768, H/8, W/8]
        x = self.stem_conv(x)
        
        # Flatten patches:  [B, 768, H/8, W/8] -> [B, 768, num_patches] -> [B, num_patches, 768]
        x = x.flatten(2).transpose(1, 2)
        
        # Layer norm
        x = self.stem_ln(x)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Output layer norm
        x = self.output_ln(x)
        
        return x  # [B, num_patches, 768]
    
    def get_pooled_features(self, x):
        """Get global average pooled features."""
        features = self.forward(x)  # [B, num_patches, 768]
        pooled = features.mean(dim=1)  # [B, 768]
        return pooled
    
    def load_pretrained_weights(self, weights_path):
        """Load pretrained weights (supports .pth format)."""
        
        if weights_path.endswith('.pth') or weights_path.endswith('.pt'):
            # Load PyTorch checkpoint directly
            state_dict = torch.load(weights_path, map_location='cpu')
            print(f"\n{'='*80}")
            print(f"Loading pretrained weights from: {weights_path}")
            print(f"{'='*80}")
            
        else:
            raise ValueError(f"Unsupported format: {weights_path}. Please use .pth format.")
        
        # Load state dict
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        
        print(f"\n✓ Loaded pretrained weights")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:3]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
        print(f"{'='*80}\n")
    
    def _convert_tf_weights_to_pytorch(self, tf_weights: Dict) -> Dict:
        """Convert TensorFlow weights to PyTorch format."""
        print("Converting TensorFlow weights to PyTorch...")
        
        pytorch_state_dict = {}
        converted_count = 0
        
        for tf_name, tf_tensor in tf_weights.items():
            # Convert to torch
            tensor = torch.from_numpy(tf_tensor)
            
            # Parse TF name
            pt_name = self._map_tf_name_to_pytorch(tf_name, tensor)
            
            if pt_name: 
                pytorch_state_dict[pt_name] = tensor
                converted_count += 1
                if converted_count <= 10 or converted_count % 20 == 0:
                    # Only print first 10 and every 20th to avoid spam
                    print(f"  [{converted_count: 3d}] {tf_name} -> {pt_name}")
        
        print(f"\n✓ Converted {converted_count} / {len(tf_weights)} weights")
        return pytorch_state_dict
    
    def _map_tf_name_to_pytorch(self, tf_name: str, tensor: torch.Tensor) -> Optional[str]:
        """Map TensorFlow variable name to PyTorch parameter name."""
        
        # Stem convolution
        if 'stem_conv/kernel' in tf_name: 
            # TF: [H, W, C_in, C_out], PyTorch: [C_out, C_in, H, W]
            self._temp_tensor = tensor.permute(3, 2, 0, 1)
            return 'stem_conv.weight'
        elif 'stem_conv/bias' in tf_name:
            return 'stem_conv.bias'
        
        # Stem LayerNorm
        elif 'stem_ln/gamma' in tf_name:
            return 'stem_ln.weight'
        elif 'stem_ln/beta' in tf_name:
            return 'stem_ln.bias'
        
        # Output LayerNorm
        elif 'output_ln/gamma' in tf_name:
            return 'output_ln.weight'
        elif 'output_ln/beta' in tf_name:
            return 'output_ln.bias'
        
        # Transformer layers
        elif 'enc_layers/' in tf_name:
            # Extract layer number
            parts = tf_name.split('/')
            layer_idx = None
            for i, part in enumerate(parts):
                if 'enc_layers' in part:
                    # Next part should be the number
                    if i + 1 < len(parts) and parts[i + 1].isdigit():
                        layer_idx = int(parts[i + 1])
                        break
            
            if layer_idx is None:
                return None
            
            prefix = f'encoder_layers.{layer_idx}'
            
            # MHA LayerNorm
            if 'mha_ln/gamma' in tf_name:
                return f'{prefix}.mha_ln.weight'
            elif 'mha_ln/beta' in tf_name: 
                return f'{prefix}.mha_ln.bias'
            
            # MHA projections
            elif 'mha/_query_dense/kernel' in tf_name:
                # TF: [768, 12, 64], PyTorch: [768, 768] needs reshape
                self._temp_tensor = tensor.reshape(self.hidden_dim, -1).t()
                return f'{prefix}.mha.query_dense.weight'
            elif 'mha/_query_dense/bias' in tf_name:
                # TF: [12, 64], PyTorch: [768]
                self._temp_tensor = tensor.flatten()
                return f'{prefix}.mha.query_dense.bias'
            
            elif 'mha/_key_dense/kernel' in tf_name:
                self._temp_tensor = tensor.reshape(self.hidden_dim, -1).t()
                return f'{prefix}.mha.key_dense.weight'
            elif 'mha/_key_dense/bias' in tf_name: 
                self._temp_tensor = tensor.flatten()
                return f'{prefix}.mha.key_dense.bias'
            
            elif 'mha/_value_dense/kernel' in tf_name: 
                self._temp_tensor = tensor.reshape(self.hidden_dim, -1).t()
                return f'{prefix}.mha.value_dense.weight'
            elif 'mha/_value_dense/bias' in tf_name:
                self._temp_tensor = tensor.flatten()
                return f'{prefix}.mha.value_dense.bias'
            
            elif 'mha/_output_dense/kernel' in tf_name: 
                # TF: [12, 64, 768], PyTorch: [768, 768]
                self._temp_tensor = tensor.reshape(-1, self.hidden_dim).t()
                return f'{prefix}.mha.output_dense.weight'
            elif 'mha/_output_dense/bias' in tf_name:
                return f'{prefix}.mha.output_dense.bias'
            
            # MLP
            elif 'mlp/mlp_layers/0/dense1/kernel' in tf_name: 
                # TF: [768, 3072], PyTorch: [3072, 768]
                self._temp_tensor = tensor.t()
                return f'{prefix}.mlp.dense1.weight'
            elif 'mlp/mlp_layers/0/dense1/bias' in tf_name:
                return f'{prefix}.mlp.dense1.bias'
            
            elif 'mlp/mlp_layers/0/dense2/kernel' in tf_name:
                # TF: [3072, 768], PyTorch: [768, 3072]
                self._temp_tensor = tensor.t()
                return f'{prefix}.mlp.dense2.weight'
            elif 'mlp/mlp_layers/0/dense2/bias' in tf_name: 
                return f'{prefix}.mlp.dense2.bias'
            
            # MLP LayerNorm
            elif 'mlp/layernorms/0/gamma' in tf_name:
                return f'{prefix}.mlp.layernorm.weight'
            elif 'mlp/layernorms/0/beta' in tf_name:
                return f'{prefix}.mlp.layernorm.bias'
        
        return None


# ============================================================================
# CLASSIFICATION HEAD
# ============================================================================

class ClassificationHead(nn.Module):
    """Classification head for downstream tasks."""
    
    def __init__(self,
                 input_dim=6912,
                 num_classes=6,
                 hidden_dims=[32],
                 dropout=0.1):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        # Final classification layer
        layers.append(nn.Linear(dims[-1], num_classes))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.head(x)


# ============================================================================
# COMPLETE MODEL
# ============================================================================

class EncoderClassifier(nn.Module):
    def __init__(self,
                 num_classes=6,
                 num_channels=9,
                 pretrained_encoder_path=None,
                 freeze_encoder=True,
                 hidden_dims=[32],
                 dropout=0.1,
                 encoder_dim=768,
                 patch_size=8,
                 image_size=224,
                 use_attention_pooling=True):  # ← Add this parameter
        super().__init__()
        
        self.num_channels = num_channels
        self.encoder_dim = encoder_dim
        
        # Encoder
        self.encoder = Pix2SeqViTEncoder(
            patch_size=patch_size,
            image_size=image_size,
            pretrained_weights_path=pretrained_encoder_path
        )
        
        # Freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("✓ Encoder frozen")
        
        # ← ADD ATTENTION POOLING
        if use_attention_pooling:
            self.pooling = AttentionPooling(hidden_dim=encoder_dim, num_heads=1)
            print("✓ Using attention pooling")
        else:
            self.pooling = lambda x: x.mean(dim=1)  # Default mean pooling
            print("✓ Using mean pooling")
        
        # Classifier
        flatten_input_dim = num_channels * encoder_dim
        self.classifier = ClassificationHead(
            input_dim=flatten_input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        self._print_summary(freeze_encoder, hidden_dims, num_classes)
    
    def forward(self, x):
        # [B*C, 3, 224, 224] -> [B*C, num_patches, 768]
        features = self.encoder(x)
        
        # ← USE ATTENTION POOLING INSTEAD OF MEAN
        # [B*C, num_patches, 768] -> [B*C, 768]
        pooled = self.pooling(features)
        
        # [B*C, 768] -> [B, C, 768]
        batch_size = pooled.size(0) // self.num_channels
        pooled = pooled.view(batch_size, self.num_channels, -1)
        
        # [B, C, 768] -> [B, C*768]
        flattened = pooled.view(batch_size, -1)
        
        # [B, C*768] -> [B, num_classes]
        logits = self.classifier(flattened)
        
        return logits

    def _print_summary(self, freeze_encoder, hidden_dims, num_classes):
        print(f"\n=== EncoderClassifier (PatchTST-style) ===")
        print(f"Encoder: ViT-Base ({self.encoder_dim} dim, 12 layers)")
        print(f"Channels: {self.num_channels}")
        print(f"Flattened input: {self.num_channels * self.encoder_dim}")
        print(f"Frozen: {freeze_encoder}")
        print(f"Head: {hidden_dims}")
        print(f"Classes: {num_classes}")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")