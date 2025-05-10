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
        self.input_length = config['input_length']
        self.pred_length = config['trainging_pred_length']

    def load(self, file_name):
        I = Image.open(file_name)
        I_array = np.array(I)[:,:,0]
        
        return I_array


    def getPath(self, index):

        id_num = int(index/400)
        filePath = self.data_path

        return filePath+str(id_num)+'/'
    
    def grid_emb(self,x):

        mask = torch.ones_like(x)
        maskd0 = mask.clone()
        mask[0,:] = 0
        mask[self.img_height-1,:] = 0
        mask[:,0] = 0
        mask[:,self.img_width-1] = 0

        gridx = torch.tensor(np.linspace(0, 1, self.img_width), dtype=torch.float)
        gridy = torch.tensor(np.linspace(0, 1, self.img_height), dtype=torch.float)
        gridx = gridx.reshape(1, self.img_width).repeat([self.img_height ,1])
        gridy = gridy.reshape(self.img_height, 1).repeat([1, self.img_width])

        maskd = skfmm.distance(mask, dx = 1)
        maskd = torch.FloatTensor(maskd)

        maskd1 = torch.where(maskd<1.5,0,1)
        maskd2 = torch.where(maskd<2.5,0,1)

        grid = torch.stack([gridx,gridy,mask,maskd/256],0) # C H W 
        

        return grid, maskd0, maskd1, maskd2


    def get_dataset_data(self, index):
        base = 100
        filePath = self.getPath(index)
        index = index % 300

        pth = os.listdir(filePath)
        pth.sort()

        x_data = []
        y_data = []

        x_data_0 = torch.Tensor(self.load(filePath+pth[base]))
        
        for i in range(0, self.input_length): 
            x_file_name = filePath+pth[i+index+base]
            x_data_i = self.load(x_file_name)

            x_data.append(torch.Tensor(x_data_i))

        for i in range(self.input_length,self.input_length+self.pred_length):
            y_file_name = filePath+pth[i+index+base]
            y_data_i = self.load(y_file_name)
        
            y_data.append(torch.Tensor(y_data_i))
        
        X_data = torch.stack(x_data,-1) # H W T
        Y_data = torch.stack(y_data,-1) # H W T

        grid, maskd0, maskd1, maskd2 = self.grid_emb(x_data_0)

        X_data = X_data.unsqueeze(0)/255 # C H W T
        Y_data = Y_data.unsqueeze(0)/255 # C H W T

        grid = grid.repeat(self.input_length,1,1,1)
        grid = grid.permute(1,2,3,0)

        return X_data, Y_data, grid, maskd0, maskd1, maskd2

    def __getitem__(self, index):

        X_data, Y_data, grid, maskd0, maskd1, maskd2 = self.get_dataset_data(index)
        return X_data, Y_data, grid, maskd0, maskd1, maskd2
    
    def __len__(self):
        return 1500
        

def make_dataloader(configs):
    flow_dataset = FlowDataset(configs)
    data_loader = DataLoader(flow_dataset, batch_size= 1, shuffle= True, drop_last= True)

    return data_loader






