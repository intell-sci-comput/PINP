import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class DoubleConv (nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 3):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.BatchNorm3d(in_channels), # N C H W
            nn.ReLU(),
            spectral_norm(nn.Conv3d(in_channels, out_channels,kernel_size= kernel, padding=kernel//2)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            spectral_norm(nn.Conv3d(out_channels, out_channels,kernel_size= kernel, padding=kernel//2)),
        )
        self.single_conv = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size= kernel, padding=kernel//2))
        )

    def forward(self,x):

        single = self.single_conv(x)
        double = self.double_conv(x)

        return single + double

class DoubleConv2d (nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 3):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels), # N C H W
            nn.ReLU(),
            spectral_norm(nn.Conv2d(in_channels, out_channels,kernel_size= kernel, padding=kernel//2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channels, out_channels,kernel_size= kernel, padding=kernel//2)),
        )
        self.single_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size= kernel, padding=kernel//2))
        )

    def forward(self,x):

        single = self.single_conv(x)
        double = self.double_conv(x)

        return single + double
    
class Down (nn.Module):

    def __init__(self, in_channels, out_channels, kernel = 3):

        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((2,2,1),(2,2,1)),
            DoubleConv(in_channels, out_channels, kernel)
        )
    
    def forward(self,x):

        x = self.maxpool_conv(x)

        return x 

class Up (nn.Module):

    def __init__(self, in_channels, out_channels, kernel = 3):

        super().__init__()

        self.up = nn.Sequential(
        nn.Upsample(scale_factor=((2,2,1))),
        DoubleConv(in_channels, out_channels, kernel)
        )
        self.conv = DoubleConv(in_channels, out_channels, kernel)
    
    def forward(self, x1, x2):

        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)
    
class Encoder(nn.Module):
    def __init__(self, n_channels = 1, base_c = 64, time = 4, out_channels = 1):
        super().__init__()
        self.n_channels = n_channels
        self.base_c = base_c
        
        self.inc = DoubleConv(self.base_c//2,self.base_c) # n_channels = 1 灰度图
        self.down1 = Down(self.base_c*1, self.base_c*2)
        self.down2 = Down(self.base_c*2, self.base_c*4)
        self.down3 = Down(self.base_c*4, self.base_c*8)
        self.down4 = Down(self.base_c*8, self.base_c*16)

        self.up1 = Up(self.base_c*16, self.base_c*8)
        self.up2 = Up(self.base_c*8, self.base_c*4)
        self.up3 = Up(self.base_c*4, self.base_c*2)
        self.up4 = Up(self.base_c*2, self.base_c)

        self.fin = DoubleConv(self.base_c, self.base_c//2)
        self.l1 = nn.Linear(self.n_channels,self.base_c//2)
        self.l2 = nn.Linear(self.base_c//2, out_channels)

    def forward (self, x):
        x = self.l1(x)
        x = x.permute(0,4,1,2,3) # B C H W L

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.fin(x) # B C H W L

        x = x.permute(0,2,3,4,1)

        x = self.l2(x) # B H W L 1

        return x

    
class FlowNet(nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.configs = configs
        self.C1 = Encoder(9,configs['base_c'],4,1)
        
        self.V1 = Encoder(9,configs['base_c'],4,3)
        self.F1 = Encoder(9,configs['base_c'],4,3)
        
        self.P1 = Encoder(9,configs['base_c'],4,1)

        self.Pe = torch.nn.Parameter(torch.FloatTensor([0.001]), requires_grad=True)
        self.Re = torch.nn.Parameter(torch.FloatTensor([0.001]), requires_grad=True)

        self.fix = Encoder(5,configs['base_c'],4,1)
    
    def forward(self,X):

        C = self.C1(X) 
        V = self.V1(X) 
        P = self.P1(X)
        F1 = self.F1(X)
    
        return C,V,P,F1, self.Pe, self.Re
    
    def predict(self,C,V, Pe, X_data, maskd1, maskd2): # B C H W T
        Vy = V[:,:,:,:,0:1]
        Vx = V[:,:,:,:,1:2]
        Vz = V[:,:,:,:,2:3]

        dCdx = torch.gradient(C,dim=2)[0] 
        dCdy = torch.gradient(C,dim=1)[0]
        dCdz = torch.gradient(C,dim=3)[0]
        ddCddx = torch.gradient(dCdx,dim=2)[0] 
        ddCddy = torch.gradient(dCdy,dim=1)[0] 
        ddCddz = torch.gradient(dCdz,dim=3)[0] 

        dCdx = dCdx 
        dCdy = dCdy 
        dCdz = dCdz 

        X_pre = X_data[:,:,:,:,-1:] - Vx * dCdx - Vy * dCdy - Vz * dCdz + Pe*(ddCddx + ddCddy + ddCddz)

        return X_pre
    
    def Fix(self,X):

        return self.fix(X)




