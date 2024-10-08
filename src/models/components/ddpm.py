"""
UNet Architecture with Sinusoidal Time Embedding

This module defines a UNet-based neural network architecture enhanced with sinusoidal time embeddings.
The model is suitable for tasks requiring temporal information integration, such as time-series segmentation
or conditional image generation.

Classes:
    SinusoidalTimeEmbedding: Generates sinusoidal embeddings for temporal information.
    ConvBlock: A convolutional block with residual connections and time embedding integration.
    UNet: The main UNet architecture integrating multiple ConvBlocks for downsampling and upsampling paths.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Generates sinusoidal embeddings for temporal information.

    This class creates embeddings based on sine and cosine functions, similar to positional encodings
    used in Transformer architectures. These embeddings can be used to incorporate temporal information
    into the model.

    Args:
        time_embed_dim (int): The dimensionality of the time embedding.
    """

    def __init__(self, time_embed_dim: int):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.time_embed_dim = time_embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate sinusoidal time embeddings.

        Args:
            t (torch.Tensor): Tensor containing time steps of shape [B], where B is the batch size.

        Returns:
            torch.Tensor: Sinusoidal embeddings of shape [B, time_embed_dim].
        """
        device = t.device
        half_dim = self.time_embed_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None] * emb[None, :]  # Shape: [B, half_dim]
        time_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # Shape: [B, time_embed_dim]
        return time_emb


class ConvBlock(nn.Module):
    """
    Convolutional Block with Residual Connections and Time Embedding Integration.

    This block consists of two convolutional layers each followed by batch normalization and ReLU activation.
    Additionally, it integrates a time embedding into the convolutional layers to incorporate temporal information.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        time_emb_dim (int): Dimensionality of the time embedding.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ConvBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [B, in_channels, H, W].
            t_emb (torch.Tensor): Time embedding tensor of shape [B, time_emb_dim].

        Returns:
            torch.Tensor: Output tensor of shape [B, out_channels, H, W].
        """
        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Integrate time embedding
        t_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)  # Shape: [B, out_channels, 1, 1]
        out = out + t_emb  # Residual connection

        # Second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    """
    UNet Architecture with Sinusoidal Time Embedding.

    The UNet model consists of an encoder (downsampling path) and a decoder (upsampling path) with skip connections.
    It integrates temporal information through sinusoidal time embeddings.

    Args:
        in_channels (int, optional): Number of input channels. Default is 1.
        out_channels (int, optional): Number of output channels. Default is 1.
        time_emb_dim (int, optional): Dimensionality of the time embedding. Default is 256.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        time_emb_dim: int = 256,
    ):
        super(UNet, self).__init__()

        # Time embedding module
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        # Downsampling path
        self.down1 = ConvBlock(in_channels, 64, time_emb_dim)
        self.down2 = ConvBlock(64, 128, time_emb_dim)
        self.down3 = ConvBlock(128, 256, time_emb_dim)
        self.down4 = ConvBlock(256, 512, time_emb_dim)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024, time_emb_dim)

        # Upsampling path
        self.upconv1 = nn.ConvTranspose2d(
            1024, 512, kernel_size=2, stride=2
        )  # Upsample by a factor of 2
        self.up1 = ConvBlock(512 + 512, 512, time_emb_dim)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = ConvBlock(256 + 256, 256, time_emb_dim)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = ConvBlock(128 + 128, 128, time_emb_dim)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = ConvBlock(64 + 64, 64, time_emb_dim)

        # Final convolution to produce the output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the UNet model.

        Args:
            x (torch.Tensor): Input tensor of shape [B, in_channels, H, W].
            t (torch.Tensor): Tensor containing time steps of shape [B].

        Returns:
            torch.Tensor: Output tensor of shape [B, out_channels, H, W].
        """
        # Generate time embeddings
        t_emb = self.time_embedding(t)  # Shape: [B, time_emb_dim]

        # Encoder: Downsampling path
        d1 = self.down1(x, t_emb)               # Shape: [B, 64, H, W]
        d2 = self.down2(self.pool(d1), t_emb)   # Shape: [B, 128, H/2, W/2]
        d3 = self.down3(self.pool(d2), t_emb)   # Shape: [B, 256, H/4, W/4]
        d4 = self.down4(self.pool(d3), t_emb)   # Shape: [B, 512, H/8, W/8]

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(d4), t_emb)  # Shape: [B, 1024, H/16, W/16]

        # Decoder: Upsampling path with skip connections
        u1 = self.upconv1(bottleneck)               # Shape: [B, 512, H/8, W/8]
        u1 = torch.cat([u1, d4], dim=1)             # Concatenate along channel axis
        u1 = self.up1(u1, t_emb)                    # Shape: [B, 512, H/8, W/8]

        u2 = self.upconv2(u1)                       # Shape: [B, 256, H/4, W/4]
        u2 = torch.cat([u2, d3], dim=1)
        u2 = self.up2(u2, t_emb)                    # Shape: [B, 256, H/4, W/4]

        u3 = self.upconv3(u2)                       # Shape: [B, 128, H/2, W/2]
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.up3(u3, t_emb)                    # Shape: [B, 128, H/2, W/2]

        u4 = self.upconv4(u3)                       # Shape: [B, 64, H, W]
        u4 = torch.cat([u4, d1], dim=1)
        u4 = self.up4(u4, t_emb)                    # Shape: [B, 64, H, W]

        # Final output layer
        out = self.final_conv(u4)                   # Shape: [B, out_channels, H, W]
        return out
