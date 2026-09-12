"""Alternative auxiliary-task baselines for the paper's ablation (claim C1).

The flagship model uses *difficulty-structured class-decomposition* auxiliary
tasks. This file provides the controls we must beat:

    - SingleTaskModel       no auxiliaries — single main-class head
    - ReconAuxModel         main head + reconstruction auxiliary head
                            (same MTL weighting, same encoder)
    - MetricAuxModel        main head + contrastive metric-learning auxiliary
                            (SimCLR-style NT-Xent on encoder features)

All three share the same `EEGEncoder` + optional `TransformerRefinement` as the
flagship, so the only difference is the auxiliary signal. This makes the
ablation a controlled test of whether class-hierarchy is the right auxiliary.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'EEGNets'))
from MultiDecoderEEG import EEGEncoder, DecoderHead  # noqa: E402
from MultiDecoder_imply import TransformerRefinement  # noqa: E402


class SingleTaskModel(nn.Module):
    """Encoder + single decoder for the main class count only.

    Ablates the contribution of multi-task learning itself (C1 baseline).
    """

    def __init__(self, n_chan, n_cls, F_T=64, K_T=3, L=2,
                 use_transformer=True, n_heads=4, n_transformer_layers=2,
                 hidden_dim=128, dropout=0.5):
        super().__init__()
        self.encoder = EEGEncoder(n_chan=n_chan, F_T=F_T, K_T=K_T, L=L)
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = TransformerRefinement(
                embed_dim=F_T, num_heads=n_heads,
                num_layers=n_transformer_layers, dropout=dropout)
        self.head = DecoderHead(in_channels=F_T, n_cls=n_cls,
                                hidden_dim=hidden_dim, dropout=dropout)
        self.main_task_name = 'main'

    def forward(self, x):
        f = self.encoder(x)
        if self.use_transformer:
            f = f + self.transformer(f)
        logits = self.head(f)
        return {'main': logits}


class ReconstructionHead(nn.Module):
    """Decoder that reconstructs the input from encoder features.

    Implemented as a minimal transposed-conv stack mirroring the encoder's
    downsampling. Outputs shape (B, 1, n_chan, T_input).
    """

    def __init__(self, in_channels, n_chan, T_input):
        super().__init__()
        self.n_chan = n_chan
        self.T_input = T_input
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
        )
        self.spatial_up = nn.Conv2d(1, 1, kernel_size=(1, 1))

    def forward(self, features, target_shape):
        x = self.up(features)
        x = F.interpolate(x, size=target_shape, mode='bilinear', align_corners=False)
        x = self.spatial_up(x)
        return x


class ReconAuxModel(nn.Module):
    """Shared encoder + main classification head + reconstruction auxiliary head.

    The auxiliary reconstruction head is a widely-used self-supervised signal
    in EEG MTL (e.g., EEGEncoder 2024). Comparing against this tests whether
    the difficulty-structured class-decomposition aux is better than the
    generic reconstruction aux.
    """

    def __init__(self, n_chan, n_cls, T_input=None, F_T=64, K_T=3, L=2,
                 use_transformer=True, n_heads=4, n_transformer_layers=2,
                 hidden_dim=128, dropout=0.5):
        super().__init__()
        self.encoder = EEGEncoder(n_chan=n_chan, F_T=F_T, K_T=K_T, L=L)
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = TransformerRefinement(
                embed_dim=F_T, num_heads=n_heads,
                num_layers=n_transformer_layers, dropout=dropout)
        self.main_head = DecoderHead(in_channels=F_T, n_cls=n_cls,
                                     hidden_dim=hidden_dim, dropout=dropout)
        self.recon_head = ReconstructionHead(
            in_channels=F_T, n_chan=n_chan, T_input=T_input or 1000)
        self.main_task_name = 'main'

    def forward(self, x):
        f = self.encoder(x)
        if self.use_transformer:
            f = f + self.transformer(f)
        logits = self.main_head(f)
        B, _, Cn, T = x.shape
        recon = self.recon_head(f, target_shape=(Cn, T))
        return {'main': logits, 'recon': recon, 'recon_target': x}


class MetricAuxModel(nn.Module):
    """Shared encoder + main head + contrastive metric-learning auxiliary.

    The auxiliary produces a projection z of encoder features and applies
    NT-Xent loss treating same-class trials as positives (supervised
    contrastive, Khosla et al. 2020). Tests whether SSL-style auxiliaries
    match or beat difficulty-structured ones.
    """

    def __init__(self, n_chan, n_cls, F_T=64, K_T=3, L=2,
                 use_transformer=True, n_heads=4, n_transformer_layers=2,
                 hidden_dim=128, dropout=0.5, proj_dim=64):
        super().__init__()
        self.encoder = EEGEncoder(n_chan=n_chan, F_T=F_T, K_T=K_T, L=L)
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = TransformerRefinement(
                embed_dim=F_T, num_heads=n_heads,
                num_layers=n_transformer_layers, dropout=dropout)
        self.main_head = DecoderHead(in_channels=F_T, n_cls=n_cls,
                                     hidden_dim=hidden_dim, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(F_T, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.main_task_name = 'main'

    def forward(self, x):
        f = self.encoder(x)
        if self.use_transformer:
            f = f + self.transformer(f)
        logits = self.main_head(f)
        z = self.pool(f).flatten(1)
        z = self.proj(z)
        z = F.normalize(z, dim=1)
        return {'main': logits, 'metric_z': z}


def supcon_loss(z, labels, temperature=0.1):
    """Supervised contrastive loss (Khosla et al. 2020).

    Args:
        z: (B, D) L2-normalized projections.
        labels: (B,) class labels.
    """
    B = z.shape[0]
    sim = z @ z.t() / temperature
    # Mask out self-similarity
    mask_self = torch.eye(B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask_self, -1e9)
    # Positive mask: same label, not self
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()).float()
    pos_mask.masked_fill_(mask_self, 0.0)

    # log-softmax over all non-self
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
    return -mean_log_prob_pos.mean()


def build_aux_model(aux_type, n_chan, n_cls, T_input=None, **model_kwargs):
    """Factory for auxiliary-task baselines."""
    aux_type = aux_type.lower()
    if aux_type == 'single':
        return SingleTaskModel(n_chan=n_chan, n_cls=n_cls, **model_kwargs)
    elif aux_type == 'recon':
        return ReconAuxModel(n_chan=n_chan, n_cls=n_cls,
                             T_input=T_input, **model_kwargs)
    elif aux_type == 'metric':
        return MetricAuxModel(n_chan=n_chan, n_cls=n_cls, **model_kwargs)
    else:
        raise ValueError(f"Unknown aux_type: {aux_type}")
