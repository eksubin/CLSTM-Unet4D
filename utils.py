import torch
import torch.nn as nn
import torch.nn.functional as F

class CLSTMCell3D(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(CLSTMCell3D, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # to preserve spatial dimensions
        self.conv = nn.Conv3d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding
        )
    
    def forward(self, x, h, c):
        # Concatenate input and hidden state along the channel dimension
        combined = torch.cat([x, h], dim=1)
        conv_output = self.conv(combined)
        i, f, o, g = torch.chunk(conv_output, 4, dim=1)
        
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        o = torch.sigmoid(o)  # output gate
        g = torch.tanh(g)     # cell gate
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class DoubleConv3D(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """Downscaling with maxpool followed by DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Down3D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling followed by DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up3D, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust dimensions if necessary
        diff_depth = x2.size(2) - x1.size(2)
        diff_height = x2.size(3) - x1.size(3)
        diff_width = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diff_width // 2, diff_width - diff_width // 2,
                        diff_height // 2, diff_height - diff_height // 2,
                        diff_depth // 2, diff_depth - diff_depth // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    """Final 1x1 convolution"""
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# CLSTM3D (reusing from previous response)
class CLSTM3D(nn.Module):
    """ConvLSTM module for 3D data."""
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True, batch_first=True):
        super(CLSTM3D, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.lstm_cells = nn.ModuleList([ 
            CLSTMCell3D(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])

    def forward(self, x):
        # Initialize hidden and cell states
        b, t, c, d, h, w = x.size()
        hidden_states = [None] * self.num_layers
        outputs = []
        
        for t_step in range(t):
            x_t = x[:, t_step] if self.batch_first else x[t_step]
            for layer_idx, lstm_cell in enumerate(self.lstm_cells):
                if hidden_states[layer_idx] is None:
                    hidden_states[layer_idx] = (torch.zeros(b, self.hidden_dim, d, h, w).to(x.device),
                                                torch.zeros(b, self.hidden_dim, d, h, w).to(x.device))
                hidden_state, cell_state = lstm_cell(x_t, hidden_states[layer_idx][0], hidden_states[layer_idx][1])
                hidden_states[layer_idx] = (hidden_state, cell_state)
                x_t = hidden_state
            outputs.append(x_t.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)

