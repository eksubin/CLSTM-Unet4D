import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

# U-Net 3D with CLSTM bottleneck (for time sequence)
class UNet3DWithTime(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet3DWithTime, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        self.clstm = CLSTM3D(512, 512, kernel_size=3, num_layers=1)  # Bottleneck with CLSTM3D
        self.up1 = Up3D(768, 256, bilinear)  # Adjust the channels here
        self.up2 = Up3D(384, 128, bilinear)
        self.up3 = Up3D(192, 64, bilinear)
        self.up4 = Up3D(64 + 64, 64, bilinear)
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x):
        b, t, c, d, h, w = x.size()  # Expecting 6D input (batch, time, channels, depth, height, width)
        x = x.view(b * t, c, d, h, w)  # Merge batch and time for initial encoding
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4 = x4.view(b, t, -1, x4.size(2), x4.size(3), x4.size(4))  # Reshape for ConvLSTM
        x4 = self.clstm(x4)  # Apply CLSTM3D
        x4 = x4.contiguous().view(b * t, -1, x4.size(3), x4.size(4), x4.size(5))  # Reshape back

        x = self.up1(x4, x3)  # First upsampling with concatenation
        x = self.up2(x, x2)   # Second upsampling
        x = self.up3(x, x1)   # Third upsampling
        x = self.up4(x, x1)   # Fourth upsampling
        logits = self.outc(x)  # Final output
        logits = logits.view(b, t, -1, logits.size(2), logits.size(3), logits.size(4))  # Reshape to include time
        return logits
        # added sigmoid before thelogits
