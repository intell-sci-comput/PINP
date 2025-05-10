import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=kernel // 2)),
        )
        self.single_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2))
        )

    def forward(self, x):
        return self.single_conv(x) + self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.up = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        DoubleConv(in_channels, out_channels, kernel)
        )
        self.conv = DoubleConv(in_channels, out_channels, kernel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, n_channels=1, base_c=64, time=4, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)

        self.up1 = Up(base_c * 16, base_c * 8)
        self.up2 = Up(base_c * 8, base_c * 4)
        self.up3 = Up(base_c * 4, base_c * 2)
        self.up4 = Up(base_c * 2, base_c)

        self.fin = DoubleConv(base_c, out_channels)
        self.fct = nn.Linear(time, 1)

    def forward(self, x):  # x: B C H W T
        B, C, H, W, T = x.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(B, T* C, H, W)  # -> (B*T) C H W

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.fin(x)  # shape: (B*T, C_out, H, W)

        _, Cout, H, W = x.shape
        x = x.view(B, T, Cout//T, H, W).permute(0, 2, 3, 4, 1)  # -> (B, Cout, H, W, T)

        x = self.fct(x)  # shape: (B, Cout, H, W, 1)
        return x

class FlowNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.C1 = Encoder(20, configs['base_c'], 4, 4)
        self.V1 = Encoder(20, configs['base_c'], 4, 8)
        self.P1 = Encoder(20, configs['base_c'], 4, 4)

        self.Pe = nn.Parameter(torch.tensor([0.001], dtype=torch.float32), requires_grad=True)
        self.Re = nn.Parameter(torch.tensor([0.001], dtype=torch.float32), requires_grad=True)

        self.fix = Encoder(5, configs['base_c'], 5, 5)

    def forward(self, X):
        C = self.C1(X)
        V = self.V1(X)
        P = self.P1(X)
        return C, V, P, self.Pe, self.Re

    def predict(self, C, V, Pe, X_data, maskd1, maskd2):
        Vy = V[:, :1, :, :, :]
        Vx = V[:, 1:, :, :, :]

        dCdx = torch.gradient(C, dim=3)[0]
        dCdy = torch.gradient(C, dim=2)[0]
        ddCddx = torch.gradient(dCdx, dim=3)[0] * maskd2[0]
        ddCddy = torch.gradient(dCdy, dim=2)[0] * maskd2[0]
        dCdx = dCdx * maskd1
        dCdy = dCdy * maskd1

        X_pre = X_data[:, :, :, :, -1:] - Vx * dCdx - Vy * dCdy + Pe * (ddCddx + ddCddy)
        return X_pre

    def Fix(self, X):
        return self.fix(X)
