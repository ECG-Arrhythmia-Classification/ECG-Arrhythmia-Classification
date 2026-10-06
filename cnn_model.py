"""
CNN Model for ECG Arrhythmia Classification (Person 3).

This module implements a 1D Convolutional Neural Network with residual connections
and dual-pooling (Adaptive Average Pooling + Adaptive Max Pooling) designed
for 5-class ECG heartbeat classification (AAMI standard: N, S, V, F, Q).
"""

from typing import Tuple
import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """Basic 1D Convolution block with BatchNorm, ReLU, and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with two convolutional layers and a shortcut projection.
    Helps smooth gradient propagation across ECG feature scales.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.15,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out + residual)
        return out


class ECGCNN(nn.Module):
    """
    1D Convolutional Neural Network for ECG Heartbeat Arrhythmia Classification.

    Architecture:
      - Stem: 1D Conv (kernel 7) + BN + ReLU + MaxPool1d
      - Stage 1: ResBlock (32 -> 32, k=5) + MaxPool1d
      - Stage 2: ResBlock (32 -> 64, k=5) + MaxPool1d
      - Stage 3: ResBlock (64 -> 128, k=3) + MaxPool1d
      - Stage 4: ResBlock (128 -> 128, k=3)
      - Pooling: Concatenation of Global Avg Pool and Global Max Pool (256 dims)
      - Classifier: Linear(256, 128) -> BN -> ReLU -> Dropout -> Linear(128, num_classes)

    Input:
      - Shape: (batch_size, 180) or (batch_size, 1, 180)
    Output:
      - Logits of shape: (batch_size, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        input_length: int = 180,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length

        # Stem: capture broad morphological features from raw/preprocessed ECG
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels, 32, kernel_size=7, stride=1, padding=3, bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 180 -> 90
        )

        # Convolutional stages with residual blocks
        self.stage1 = nn.Sequential(
            ResidualBlock1D(32, 32, kernel_size=5, dropout=dropout * 0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 90 -> 45
        )

        self.stage2 = nn.Sequential(
            ResidualBlock1D(32, 64, kernel_size=5, dropout=dropout * 0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 45 -> 22
        )

        self.stage3 = nn.Sequential(
            ResidualBlock1D(64, 128, kernel_size=3, dropout=dropout * 0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 22 -> 11
        )

        self.stage4 = nn.Sequential(
            ResidualBlock1D(128, 128, kernel_size=3, dropout=dropout * 0.5),
        )

        # Dual Global Pooling: Average + Max pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Classification Head
        feature_dim = 128 * 2  # 128 avg + 128 max
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming normal initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract high-level 256-d feature representation before classifier."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 180) -> (B, 1, 180)

        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        avg_pool = self.global_avg_pool(out).flatten(1)
        max_pool = self.global_max_pool(out).flatten(1)
        features = torch.cat([avg_pool, max_pool], dim=1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_cnn_model(
    in_channels: int = 1,
    num_classes: int = 5,
    input_length: int = 180,
    dropout: float = 0.3,
) -> ECGCNN:
    """Factory helper to instantiate ECGCNN model."""
    return ECGCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        input_length=input_length,
        dropout=dropout,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("ECG CNN Model Sanity Check")
    print("=" * 60)

    model = build_cnn_model()
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test with 2D tensor (batch_size=8, 180)
    dummy_input_2d = torch.randn(8, 180)
    out_2d = model(dummy_input_2d)
    print(f"Input shape (2D): {dummy_input_2d.shape} -> Output logits shape: {out_2d.shape}")
    assert out_2d.shape == (8, 5), f"Expected shape (8, 5), got {out_2d.shape}"

    # Test with 3D tensor (batch_size=8, 1, 180)
    dummy_input_3d = torch.randn(8, 1, 180)
    out_3d = model(dummy_input_3d)
    print(f"Input shape (3D): {dummy_input_3d.shape} -> Output logits shape: {out_3d.shape}")
    assert out_3d.shape == (8, 5), f"Expected shape (8, 5), got {out_3d.shape}"

    # Test backward pass
    dummy_target = torch.randint(0, 5, (8,))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out_3d, dummy_target)
    loss.backward()
    print(f"Dummy loss: {loss.item():.4f} - Backward pass successful!")
    print("All sanity checks passed successfully!")
