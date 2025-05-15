import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image


class Dataset(Dataset):
    def __init__(self, csv_path, database, loop, img_size=(224, 244)):
        column_name = ['img','mos']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_name, index_col=False, encoding='utf-8-sig')
        self.img_size = img_size
        self.X_train = tmp_df['img']
        self.Y_train = tmp_df['mos']
        self.length = len(tmp_df)
        self.database = database
        self.loop = str(loop+1)

    def __getitem__(self, index):

        # get the corresponding image index from the csv path
        if self.database == 'IQA-ODI':
            img_path = self.X_train.iloc[index]
            img_index = img_path.split('/')[6].split('.')[0]
            img_path = '/home1/mpc/Dataset/IQA-ODI/resized_imgs/' + img_index + '.png'
        else:
            img_path = self.X_train.iloc[index]
            img_index = img_path.split('/')[6].split('.')[0].rjust(3, '0')

        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize((1024, 512))
        img = transforms.ToTensor()(img)
        # load the features learned from the teacher network
        feat_path = '/home1/mpc/Learning files/SPL-36814-2023/OIQA_KD/features/' + self.database + '/' + self.loop + '/' + img_index + '.npy'
        feat = torch.load(feat_path).squeeze(0)

        y_mos = self.Y_train.iloc[index]
        y_label = torch.FloatTensor(np.array(float(y_mos)))

        return img, feat, y_label

    def __len__(self):
        return self.length