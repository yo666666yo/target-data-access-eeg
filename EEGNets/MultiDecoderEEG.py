import torch
from torch import nn
from torch.nn import functional as F


class TCN_ResidualBlock(nn.Module):
    """Temporal Convolutional Network residual block with dilated causal convolutions."""
    def __init__(self, in_chan, out_chan, K, dilation, dropout):
        super(TCN_ResidualBlock, self).__init__()
        self.dilation = dilation
        self.kernel_size = K
        self.left_pad = dilation * (K - 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(1, self.kernel_size), dilation=(1, dilation)),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.Dropout(p=dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=(1, self.kernel_size), dilation=(1, dilation)),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.Dropout(p=dropout)
        )

        if in_chan != out_chan:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1)),
                nn.BatchNorm2d(out_chan)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.pad(x, (self.left_pad, 0, 0, 0), mode='constant', value=0)
        x = self.conv1(x)
        x = F.pad(x, (self.left_pad, 0, 0, 0), mode='constant', value=0)
        x = self.conv2(x)
        return x + residual


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention for encoder features.

    Allows each decoder to selectively attend to relevant feature channels,
    enabling task-specific feature reweighting from the shared encoder output.
    """
    def __init__(self, n_chan, reduction=4):
        super(ChannelAttention, self).__init__()
        mid = max(n_chan // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(n_chan, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, n_chan, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, _, _ = x.shape
        w = self.squeeze(x).view(B, C)
        w = self.excitation(w).view(B, C, 1, 1)
        return x * w


class EEGEncoder(nn.Module):
    """Shared EEG encoder using residual depthwise-separable convolutions + TCN.

    Architecture: 4 residual conv blocks (25→50→100→200 channels) with
    depthwise-separable convolutions, followed by L layers of dilated TCN blocks.

    Input:  (Batch, 1, Channels, TimePoints)
    Output: (Batch, F_T, 1, T_compressed)
    """
    def __init__(self, n_chan, F_T=64, K_T=3, L=2):
        super(EEGEncoder, self).__init__()
        # Block 1: temporal + spatial convolution
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 11), padding=(0, 5)),
            nn.Conv2d(25, 25, kernel_size=(n_chan, 1)),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.9),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.short_1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(n_chan, 1)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(25)
        )

        # Block 2: depthwise separable conv
        self.conv_2 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=(1, 5), padding=(0, 2), groups=25, bias=False),
            nn.Conv2d(25, 50, kernel_size=1, bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.9),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.short_2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(50)
        )

        # Block 3: depthwise separable conv
        self.conv_3 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 5), padding=(0, 2), groups=50, bias=False),
            nn.Conv2d(50, 100, kernel_size=1, bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.9),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.short_3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(100)
        )

        # Block 4: depthwise separable conv
        self.conv_4 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(1, 5), padding=(0, 2), groups=100, bias=False),
            nn.Conv2d(100, 200, kernel_size=1, bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.9),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.short_4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(200)
        )

        # TCN blocks: encoder outputs 200 channels, so first TCN input is 200
        self.tcn_blocks = nn.ModuleList()
        for i in range(L):
            dilation = 2 ** i
            in_channels = 200 if i == 0 else F_T
            self.tcn_blocks.append(
                TCN_ResidualBlock(
                    in_chan=in_channels,
                    out_chan=F_T,
                    K=K_T,
                    dilation=dilation,
                    dropout=0.25
                )
            )
        self.out_channels = F_T

    def forward(self, x):
        x = F.elu(self.conv_1(x) + self.short_1(x))
        x = F.elu(self.conv_2(x) + self.short_2(x))
        x = F.elu(self.conv_3(x) + self.short_3(x))
        x = F.elu(self.conv_4(x) + self.short_4(x))
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
        return x


class DecoderHead(nn.Module):
    """Task-specific decoder head with channel attention and MLP classifier.

    Each decoder first applies channel attention to selectively reweight
    encoder features for its specific task, then uses adaptive pooling
    and a 2-layer MLP for classification.
    """
    def __init__(self, in_channels, n_cls, hidden_dim=128, dropout=0.5):
        super(DecoderHead, self).__init__()
        self.attention = ChannelAttention(in_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, n_cls)
        )

    def forward(self, x):
        x = self.attention(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class LightDecoderHead(nn.Module):
    """Lightweight decoder head without attention — for simpler sub-tasks."""
    def __init__(self, in_channels, n_cls, dropout=0.5):
        super(LightDecoderHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, n_cls),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.classifier(x)
        return x
