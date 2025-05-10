import numpy as np
import os
import shutil
from torch.utils.data import Dataset,DataLoader
import torch
# from scipy.signal import medfilt2d
from PIL import Image 
import skfmm


class FlowDataset(Dataset):

    def __init__(self, config):
    
        self.data_path = config['training_data_path']
        self.img_height = config['image_height']
        self.img_width = config['image_width']
        self.img_length = config['image_length']
        self.input_length = config['input_length']
        self.pred_length = config['trainging_pred_length']

    
    def grid_emb(self,x):

        mask = torch.ones_like(x) # H W L
        maskd0 = mask.clone()
        mask[0,:,:] = 0
        mask[self.img_height-1,:,:] = 0
        mask[:,0,:] = 0
        mask[:,self.img_width-1,:] = 0
        mask[:,:,0] = 0
        mask[:,:,self.img_length-1] = 0

        gridx = torch.tensor(np.linspace(0, 1, self.img_width), dtype=torch.float)
        gridy = torch.tensor(np.linspace(0, 1, self.img_height), dtype=torch.float)
        gridz = torch.tensor(np.linspace(0, 1, self.img_length), dtype=torch.float)

        gridx = gridx.reshape(1, self.img_width,1).repeat([self.img_height ,1, self.img_length])
        gridy = gridy.reshape(self.img_height, 1, 1).repeat([1, self.img_width, self.img_length])
        gridz = gridz.reshape(1, 1, self.img_length).repeat([self.img_height, self.img_width, 1])

        maskd = skfmm.distance(mask, dx = 1)
        maskd = torch.FloatTensor(maskd)

        maskd1 = torch.where(maskd<1.5,0,1)
        maskd2 = torch.where(maskd<2.5,0,1)

        grid = torch.stack([gridx,gridy,gridz,mask,maskd/self.img_height],-1) # H W L C  

        maskd0 = maskd0.unsqueeze(-1)
        maskd1 = maskd1.unsqueeze(-1)
        maskd2 = maskd2.unsqueeze(-1)

        return grid, maskd0, maskd1, maskd2


    def get_dataset_data(self, index):

        filePath = self.data_path
        pth = os.listdir(filePath)
        pth.sort()

        data = np.load(filePath + pth[index])

        data_c =  data['fluid_field'] # 20 64 64 64 1  T H W L C

        data_x = torch.Tensor(data_c[:self.input_length,:,:,:,:])
        data_y = torch.Tensor(data_c[self.input_length:self.input_length + self.pred_length,:,:,:,:])

        data_x = data_x.permute(4,1,2,3,0)[0]
        data_y = data_y.permute(4,1,2,3,0)[0] # H W L T


        grid, maskd0, maskd1, maskd2 = self.grid_emb(data_x[:,:,:,0])


        return data_x, data_y, grid, maskd0, maskd1, maskd2

    def __getitem__(self, index):

        X_data, Y_data, grid, maskd0, maskd1, maskd2 = self.get_dataset_data(index)
        return X_data, Y_data, grid, maskd0, maskd1, maskd2
    
    def __len__(self):
        return 330
        

def make_dataloader(configs):
    flow_dataset = FlowDataset(configs)
    data_loader = DataLoader(flow_dataset, batch_size= 1, shuffle= True, drop_last= True)

    return data_loader






