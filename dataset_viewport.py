import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image


class Dataset(Dataset):
    def __init__(self, database, csv_path, img_size=(224, 244)):
        column_name = ['img','mos']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_name, index_col=False, encoding='utf-8-sig')
        self.img_size = img_size
        self.X_train = tmp_df['img']
        self.Y_train = tmp_df['mos']
        self.length = len(tmp_df)
        self.database = database

    def __getitem__(self, index):
        
        if self.database == 'IQA-ODI':
            img_path = self.X_train.iloc[index]
            img_index = img_path.split('/')[6].split('.')[0]
            #img_index = img_path
            
        else:
            img_path = self.X_train.iloc[index]
            img_index = img_path.split('/')[6].split('.')[0].rjust(3, '0')
        
        img_path = '/home1/mpc/Dataset/' + self.database + '/resized_cubic/' + img_index
        
        img_path1 = img_path + 'F.png'
        img_path2 = img_path + 'R.png'
        img_path3 = img_path + 'BA.png'
        img_path4 = img_path + 'L.png'
        img_path5 = img_path + 'T.png'
        img_path6 = img_path + 'BO.png'
        
        img1 = Image.open(img_path1)
        img1 = img1.convert('RGB')
        img1 = img1.resize((256, 256))
        img1 = transforms.ToTensor()(img1)

        img2 = Image.open(img_path2)
        img2 = img2.convert('RGB')
        img2 = img2.resize((256, 256))
        img2 = transforms.ToTensor()(img2)

        img3 = Image.open(img_path3)
        img3 = img3.convert('RGB')
        img3 = img3.resize((256, 256))
        img3 = transforms.ToTensor()(img3)

        img4 = Image.open(img_path4)
        img4 = img4.convert('RGB')
        img4 = img4.resize((256, 256))
        img4 = transforms.ToTensor()(img4)

        img5 = Image.open(img_path5)
        img5 = img5.convert('RGB')
        img5 = img5.resize((256, 256))        
        img5 = transforms.ToTensor()(img5)

        img6 = Image.open(img_path6)
        img6 = img6.convert('RGB')
        img6 = img6.resize((256, 256))
        img6 = transforms.ToTensor()(img6)

        y_mos = self.Y_train.iloc[index]
        y_label = torch.FloatTensor(np.array(float(y_mos)))

        return img1, img2, img3, img4, img5, img6, y_label

    def __len__(self):
        return self.length