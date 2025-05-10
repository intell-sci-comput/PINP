import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class Loss (nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()

    def Main_loss(self,X_all,Y_data):

        return self.loss(X_all,Y_data)
    
    def Phy_loss(self, V_all, P_all, F_all,Re, maskd0,maskd1, maskd2):

        Vy = V_all[:,:,:,:,0,:] # B H W L T
        Vx = V_all[:,:,:,:,1,:]
        Vz = V_all[:,:,:,:,2,:]
        Fy = F_all[:,:,:,:,0,:]
        Fx = F_all[:,:,:,:,1,:]
        Fz = F_all[:,:,:,:,2,:]

        dVydt = torch.gradient(Vy,dim=4)[0]
        dVydx = torch.gradient(Vy,dim=2)[0]
        dVydy = torch.gradient(Vy,dim=1)[0]
        dVydz = torch.gradient(Vy,dim=3)[0]

        dVxdt = torch.gradient(Vx,dim=4)[0]
        dVxdx = torch.gradient(Vx,dim=2)[0]
        dVxdy = torch.gradient(Vx,dim=1)[0]
        dVxdz = torch.gradient(Vx,dim=3)[0]

        dVzdt = torch.gradient(Vz,dim=4)[0]
        dVzdx = torch.gradient(Vz,dim=2)[0]
        dVzdy = torch.gradient(Vz,dim=1)[0]
        dVzdz = torch.gradient(Vz,dim=3)[0]

        dPdx = torch.gradient(P_all,dim=2)[0]
        dPdy = torch.gradient(P_all,dim=1)[0]
        dPdz = torch.gradient(P_all,dim=3)[0]

        ddVxddx = torch.gradient(dVxdx,dim=2)[0]
        ddVxddy = torch.gradient(dVxdy,dim=1)[0]
        ddVxddz = torch.gradient(dVxdz,dim=3)[0]

        ddVyddx = torch.gradient(dVydx,dim=2)[0]
        ddVyddy = torch.gradient(dVydy,dim=1)[0]
        ddVyddz = torch.gradient(dVydz,dim=3)[0]

        ddVzddx = torch.gradient(dVzdx,dim=2)[0]
        ddVzddy = torch.gradient(dVzdy,dim=1)[0]
        ddVzddz = torch.gradient(dVzdz,dim=3)[0]

        e1 = dVydt + (Vx * dVydx + Vy * dVydy + Vz * dVydz + dPdy) - Re* (ddVyddx + ddVyddy + ddVyddz) + Fy
        e2 = dVxdt + (Vx * dVxdx + Vy * dVxdy + Vz * dVxdz + dPdx) - Re* (ddVxddx + ddVxddy + ddVxddz) + Fx
        e3 = dVzdt + (Vx * dVzdx + Vy * dVzdy + Vz * dVzdz + dPdz) - Re* (ddVzddx + ddVzddy + ddVzddz) + Fz
        e4 = (dVxdx + dVydy + dVzdz)

        
        return self.loss(e1,torch.zeros_like(e1)) + self.loss(e2,torch.zeros_like(e2)) + self.loss(e3,torch.zeros_like(e3)) + self.loss(e4,torch.zeros_like(e4)) 
    
    def Time_loss(self, C_all, Y_data, X_last,maskd0):
        Y_last = torch.cat([X_last,Y_data[:,:,:,:,:-1]],4)

        t1 = self.loss(Y_last,Y_data)
        t2 = self.loss(C_all,Y_data) 

        if t1.detach() < t2.detach():
            loss_t = t2-t1
        else:
            loss_t = 0.0
    
        return loss_t
    
    def forward(self,C_all,V_all,P_all,X_all,X1_all,F_all,Re, X_last,Y_data,maskd0,maskd1,maskd2):

       return self.Main_loss(X1_all,Y_data),self.Main_loss(X_all,Y_data), self.Phy_loss(V_all, P_all,F_all, Re, maskd0, maskd1, maskd2) ,self.Time_loss(C_all, Y_data, X_last,maskd0)



