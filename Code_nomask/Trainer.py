import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from dataset import make_dataloader

from torch.optim.lr_scheduler import StepLR
from PINP_3D import FlowNet
from loss import Loss
import matplotlib.pyplot as plt
import pandas as pd
import time
args = {
    'batch_size': 1,
    'training_data_path':'smoke_3d/',
    'image_width': 64,
    'image_height': 64,
    'image_length': 64,
    'input_length':4,
    'trainging_pred_length':4,
    'testing_pred_length':10,
    'base_c':20,
    
}  # 配置信息
cpu_num = 4 # 这里设置成你想运行的CPU个数
os.environ["OMP_NUM_THREADS"] = str(cpu_num) 
os.environ["MKL_NUM_THREADS"] = str(cpu_num) 
torch.set_num_threads(cpu_num)

configs = args
data_loader = make_dataloader(args)

n_epochs = 1000
lr = 0.001

flow_net = FlowNet(configs).cuda()

Loss_2d = Loss().cuda()
optimizer_E = torch.optim.Adam(flow_net.parameters(), lr=lr)
scheduler_E = StepLR(optimizer_E, step_size=5, gamma=0.8)

Loss_test = torch.nn.MSELoss()
Loss_test1 = torch.nn.L1Loss()

#step1:
for epoch in range(n_epochs):
    for batch_id, (X_data, Y_data, grid, maskd0, maskd1, maskd2) in enumerate(data_loader):
        
        X_data = X_data.cuda()
        Y_data = Y_data.cuda()
        grid = grid.cuda()
        maskd0, maskd1, maskd2 = maskd0.cuda(),maskd1.cuda(),maskd2.cuda()
      

        X_last = X_data[:,:,:,:,-1:]
    
        C_result = []
        V_result = []
        P_result = []
        X_result = []
        F_result = []
        X1_result = []

        for i in range(configs['trainging_pred_length']):

            X_Data = torch.cat([X_data,grid],-1) # B H W L C
            C,V,P,F1, Pe, Re = flow_net(X_Data) # B H W L C
            
            X_pre = flow_net.predict(C,V, Pe, X_data,maskd1, maskd2) # B H W L C

            X_pre1 = flow_net.Fix(torch.cat([X_data,X_pre],4))
            
            X_data = torch.cat([X_data[:,:,:,:,1:],X_pre1],4)

            C_result.append(C)
            V_result.append(V)
            P_result.append(P)
            X_result.append(X_pre)
            X1_result.append(X_pre1)
            F_result.append(F1)

        C_all = torch.cat(C_result,4)
        V_all = torch.stack(V_result,-1)
        P_all = torch.cat(P_result,4)
        X_all = torch.cat(X_result,4)
        X1_all = torch.cat(X1_result,4)
        F_all = torch.stack(F_result,-1)

        loss_m1,loss_m2, loss_p, loss_t = Loss_2d(C_all,V_all,P_all,X_all,X1_all,F_all,Re, X_last,Y_data,maskd0,maskd1,maskd2)

        loss = loss_m1 + loss_m2 + loss_p + loss_t

        optimizer_E.zero_grad()
        loss.backward()
        optimizer_E.step()

        if batch_id % 200 == 0:
            print("Epoch:{},Batch:{}".format(epoch,batch_id))
            print("loss:{}".format(loss.detach().cpu()))
            print("loss_m1:{}".format(loss_m1.detach().cpu()))
            print("loss_m2:{}".format(loss_m2.detach().cpu()))
            print("loss_p:{}".format(loss_p.detach().cpu()))
            print("loss_t:{}".format(loss.detach().cpu() - loss_m1.detach().cpu() - loss_m2.detach().cpu() - loss_p.detach().cpu()))

            print("----------------------------------------")

    scheduler_E.step()

    
        
        








        


