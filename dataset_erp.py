import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image


class Dataset(Dataset):
    def __init__(self,database, csv_path, test = False, img_size=(224, 244)):
        column_name = ['img','mos']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_name, index_col=False, encoding='utf-8-sig')
        self.img_size = img_size
        self.X_train = tmp_df['img']
        self.Y_train = tmp_df['mos']
        self.length = len(tmp_df)
        self.test = test
        self.database = database
        
    def __getitem__(self, index):
        if self.database == 'IQA-ODI':
            img_path = self.X_train.iloc[index]
            img_index = img_path.split('/')[6].split('.')[0]
            #img_index = img_path
            img_path = '/home1/mpc/Dataset/IQA-ODI/resized_imgs/' + img_index + '.png'
        else:
            img_path = self.X_train.iloc[index]
       
        img = Image.open(img_path)
        img = img.convert('RGB')

        img = img.resize((1024, 512))
        img = transforms.ToTensor()(img)

        y_mos = self.Y_train.iloc[index]
        y_mos = torch.FloatTensor(np.array(float(y_mos)))

        return img, y_mos

    def __len__(self):
        return self.length
    