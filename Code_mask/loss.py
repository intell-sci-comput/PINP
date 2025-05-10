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
    
    def Phy_loss(self, V_all, P_all, Re, maskd0, maskd1, maskd2):

        Vy = V_all[:,:1,:,:,:]
        Vx = V_all[:,1:,:,:,:]

        dVydt = torch.gradient(Vy,dim=4)[0]
        dVydx = torch.gradient(Vy,dim=3)[0]
        dVydy = torch.gradient(Vy,dim=2)[0]

        dVxdt = torch.gradient(Vx,dim=4)[0]
        dVxdx = torch.gradient(Vx,dim=3)[0]
        dVxdy = torch.gradient(Vx,dim=2)[0]

        dPdx = torch.gradient(P_all,dim=3)[0]
        dPdy = torch.gradient(P_all,dim=2)[0]

        ddVxddx = torch.gradient(dVxdx,dim=3)[0]
        ddVxddy = torch.gradient(dVxdy,dim=2)[0]
        ddVyddx = torch.gradient(dVydx,dim=3)[0]
        ddVyddy = torch.gradient(dVydy,dim=2)[0]

        e1 = dVydt*maskd0 + (Vx * dVydx + Vy * dVydy + dPdy) * maskd1 - Re * (ddVyddx + ddVyddy)* maskd2
        e2 = dVxdt*maskd0 + (Vx * dVxdx + Vy * dVxdy + dPdx) * maskd1 - Re * (ddVxddx + ddVxddy)* maskd2
        e3 = (dVxdx + dVydy)*maskd1

        return self.loss(e1,torch.zeros_like(e1)) + self.loss(e2,torch.zeros_like(e2)) + self.loss(e3,torch.zeros_like(e3))
    
    def Time_loss(self, C_all, Y_data, X_last,maskd0):
        Y_last = torch.cat([X_last,Y_data[:,:,:,:,:-1]],4)

        t1 = self.loss(Y_last,Y_data)
        t2 = self.loss(C_all*maskd0,Y_data) 

        if t1.detach() < t2.detach():
            loss_t = t2-t1
        else:
            loss_t = 0.0
    
        return loss_t
    
    def forward(self,C_all,V_all,P_all,X_all,X1_all,Re, X_last,Y_data,maskd0, maskd1,maskd2):

        return self.Main_loss(X1_all,Y_data),self.Main_loss(X_all,Y_data), self.Phy_loss(V_all, P_all, Re, maskd0, maskd1, maskd2) ,self.Time_loss(C_all, Y_data, X_last,maskd0)



