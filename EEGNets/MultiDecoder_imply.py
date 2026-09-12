import sys
import os
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MultiDecoderEEG import EEGEncoder, DecoderHead, LightDecoderHead, ChannelAttention


class MultiDecoder4Train(nn.Module):
    """Multi-decoder model for multi-task training.

    Uses a shared encoder with multiple classification heads of increasing
    difficulty. Auxiliary easy tasks (e.g., 2-class binary) provide gradient
    regularization that helps the main hard task (e.g., 5-class).

    Args:
        n_chan: Number of EEG channels
        n_cls_list: List of class counts for each decoder [easy, ..., hard]
        task_names: Optional names for each task
        F_T: TCN filter count
        K_T: TCN kernel size
        L: Number of TCN layers
        hidden_dim: Hidden dimension for decoder MLPs
        dropout: Dropout rate
    """
    def __init__(self, n_chan, n_cls_list=[2, 3, 5],
                 task_names=None, F_T=64, K_T=3, L=2,
                 hidden_dim=128, dropout=0.5):
        super(MultiDecoder4Train, self).__init__()
        self.n_cls_list = n_cls_list
        self.task_names = task_names or [f'task_{n}cls' for n in n_cls_list]

        self.encoder = EEGEncoder(n_chan=n_chan, F_T=F_T, K_T=K_T, L=L)

        self.decoders = nn.ModuleDict()
        for name, n_cls in zip(self.task_names, n_cls_list):
            if n_cls <= 3:
                self.decoders[name] = LightDecoderHead(
                    in_channels=F_T, n_cls=n_cls, dropout=dropout)
            else:
                self.decoders[name] = DecoderHead(
                    in_channels=F_T, n_cls=n_cls,
                    hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x):
        features = self.encoder(x)
        outputs = {}
        for name, decoder in self.decoders.items():
            outputs[name] = decoder(features)
        return outputs

    def get_main_output(self, x):
        """Get output from the main (hardest) task decoder only."""
        outputs = self.forward(x)
        main_task = self.task_names[-1]
        return outputs[main_task]


class MultiDecoder4Control(nn.Module):
    """Multi-decoder model for BCI robotic control.

    Separate decoders for force/speed (intensity) and direction.
    The combined control signal uses softmax probabilities from each decoder
    to produce a movement vector: intensity_prob * direction_prob.

    Args:
        n_chan: Number of EEG channels
        n_intensity: Number of intensity levels (e.g., 3: low/medium/high)
        n_direction: Number of directions (e.g., 5: up/down/left/right/stay)
        F_T: TCN filter count
        K_T: TCN kernel size
        L: Number of TCN layers
        hidden_dim: Hidden dimension for decoder MLPs
        dropout: Dropout rate
    """
    def __init__(self, n_chan, n_intensity=3, n_direction=5,
                 F_T=64, K_T=3, L=2, hidden_dim=128, dropout=0.5):
        super(MultiDecoder4Control, self).__init__()
        self.n_intensity = n_intensity
        self.n_direction = n_direction

        self.encoder = EEGEncoder(n_chan=n_chan, F_T=F_T, K_T=K_T, L=L)

        self.intensity_decoder = DecoderHead(
            in_channels=F_T, n_cls=n_intensity,
            hidden_dim=hidden_dim, dropout=dropout)

        self.direction_decoder = DecoderHead(
            in_channels=F_T, n_cls=n_direction,
            hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x):
        features = self.encoder(x)
        intensity_logits = self.intensity_decoder(features)
        direction_logits = self.direction_decoder(features)
        return {
            'intensity': intensity_logits,
            'direction': direction_logits
        }

    def get_control_signal(self, x):
        """Compute control vector from intensity and direction probabilities.

        Returns: (B, n_direction) control vector where each direction is
        weighted by the predicted intensity level.
        """
        outputs = self.forward(x)
        intensity_prob = F.softmax(outputs['intensity'], dim=-1)  # (B, n_intensity)
        direction_prob = F.softmax(outputs['direction'], dim=-1)  # (B, n_direction)
        # Intensity magnitude: weighted sum of intensity levels [0, 1, 2, ...]
        levels = torch.arange(self.n_intensity, device=x.device, dtype=x.dtype)
        magnitude = (intensity_prob * levels).sum(dim=-1, keepdim=True)  # (B, 1)
        magnitude = magnitude / (self.n_intensity - 1)  # normalize to [0, 1]
        control = magnitude * direction_prob  # (B, n_direction)
        return control


class TransformerRefinement(nn.Module):
    """Lightweight transformer refinement layer for hybrid CNN+Transformer encoder.

    Applies self-attention over the temporal dimension of CNN features to
    capture long-range dependencies that pure convolutions may miss.
    """
    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerRefinement, self).__init__()
        self.flatten_spatial = True
        self.norm_in = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout, activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, 1, T) from CNN encoder
        B, C, H, T = x.shape
        x = x.squeeze(2)          # (B, C, T)
        x = x.permute(0, 2, 1)    # (B, T, C) — sequence of feature vectors
        x = self.norm_in(x)
        x = self.transformer(x)
        x = self.norm_out(x)
        x = x.permute(0, 2, 1)    # (B, C, T)
        x = x.unsqueeze(2)        # (B, C, 1, T)
        return x


class MultiDecoderHybrid(nn.Module):
    """Hybrid deep model: CNN encoder + Transformer refinement + Multi-decoder heads.

    This is the core "hybrid deep model" that combines:
    1. Residual depthwise-separable CNN encoder (local feature extraction)
    2. TCN blocks (medium-range temporal modeling)
    3. Transformer refinement (global self-attention for long-range dependencies)
    4. Multiple task-specific decoder heads with channel attention

    Args:
        n_chan: Number of EEG channels
        n_cls_list: List of class counts for each task decoder
        task_names: Optional names for each task
        F_T: TCN/feature dimension
        K_T: TCN kernel size
        L: Number of TCN layers
        n_heads: Transformer attention heads
        n_transformer_layers: Number of transformer layers
        hidden_dim: Decoder MLP hidden dimension
        dropout: Dropout rate
        use_transformer: Whether to use transformer refinement
    """
    def __init__(self, n_chan, n_cls_list=[2, 4],
                 task_names=None, F_T=64, K_T=3, L=2,
                 n_heads=4, n_transformer_layers=2,
                 hidden_dim=128, dropout=0.5,
                 use_transformer=True):
        super(MultiDecoderHybrid, self).__init__()
        self.n_cls_list = n_cls_list
        self.task_names = task_names or [f'task_{n}cls' for n in n_cls_list]
        self.use_transformer = use_transformer

        # CNN + TCN encoder
        self.encoder = EEGEncoder(n_chan=n_chan, F_T=F_T, K_T=K_T, L=L)

        # Transformer refinement
        if use_transformer:
            self.transformer = TransformerRefinement(
                embed_dim=F_T, num_heads=n_heads,
                num_layers=n_transformer_layers, dropout=dropout
            )

        # Task-specific decoder heads
        self.decoders = nn.ModuleDict()
        for name, n_cls in zip(self.task_names, n_cls_list):
            self.decoders[name] = DecoderHead(
                in_channels=F_T, n_cls=n_cls,
                hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x):
        features = self.encoder(x)
        if self.use_transformer:
            features = features + self.transformer(features)  # residual transformer
        outputs = {}
        for name, decoder in self.decoders.items():
            outputs[name] = decoder(features)
        return outputs

    def get_main_output(self, x):
        """Get output from the main (last/hardest) task decoder only."""
        outputs = self.forward(x)
        main_task = self.task_names[-1]
        return outputs[main_task]

    def encode(self, x):
        """Get encoder + transformer features (for analysis/visualization)."""
        features = self.encoder(x)
        if self.use_transformer:
            features = features + self.transformer(features)
        return features
