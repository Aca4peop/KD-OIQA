import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image


class Dataset(Dataset):
    def __init__(self, csv_path, database, train, img_size=(224, 244)):
        column_name = ['img','mos']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_name, index_col=False, encoding='utf-8-sig')
        self.img_size = img_size
        self.X_train = tmp_df['img']
        self.Y_train = tmp_df['mos']
        self.length = len(tmp_df)
        self.database = database
        self.train = train

    def __getitem__(self, index):

        # get the corresponding image index from the csv path
        if self.database == 'IQA-ODI':
            img_path = self.X_train.iloc[index]
            img_index = img_path
            img_path = '/home1/mpc/Dataset/IQA-ODI/resized_imgs/' + img_index + '.png'
        else:
            img_path = self.X_train.iloc[index]
            img_index = img_path.split('/')[6].split('.')[0].rjust(3, '0')

        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize((1024, 512))
        img = transforms.ToTensor()(img)

        y_mos = self.Y_train.iloc[index]
        y_label = torch.FloatTensor(np.array(float(y_mos)))
        if self.train:
            # get the corresponding feature
            feat_path = '/home1/mpc/python_projects/OIQA_KD/feature_all/CVIQ-OIQA/' + img_index + '.npy'
            feat = torch.load(feat_path).squeeze(0)
            return img, feat, y_label
        else:
            return img, y_label

    def __len__(self):
        return self.length