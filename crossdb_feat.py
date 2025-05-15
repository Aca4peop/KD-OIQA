import os, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.backends.cudnn as cudnn
from policy import policy_Net
import time
import random
from tqdm import tqdm
import resnet_cmp as model_cmp
import resnet_erp as model_erp
from PIL import Image

from scipy import stats
from scipy.optimize import curve_fit


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
                        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic


def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="No reference deep 360 degree image quality assessment.")
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
                        default=100, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate.',
                        default=0.0002, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=6, type=int)
    parser.add_argument('--database_train', dest='database_train', help='The database that needs to be trained and tested.',
                        default='CVIQ', type=str)
    parser.add_argument('--database_test', dest='database_test', help='The database that needs to be trained and tested.',
                        default='OIQA', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
                        default='Mutual_model', type=str)
    parser.add_argument("--T", type=float, default=4.0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    set_seed(2023)
    s_t = time.time()

    database_train = args.database_train
    database_test = args.database_test

    teacher = model_cmp.resnet18().cuda()
    if database_train == 'CVIQ':
        if database_test == 'OIQA':
            path = '/home1/mpc/python_projects/OIQA_KD/cross_database/model_cmp_base_CVIQ_OIQA_epoch13_0.5428_0.5777_11.7418.pkl'
        elif database_test == 'IQA-ODI':
            path = '/home1/mpc/python_projects/OIQA_KD/cross_database/model_cmp_base_CVIQ_IQA-ODI_epoch14_-0.5919_0.6236_15.6763.pkl'
    elif database_train == 'OIQA':
        if database_test == 'CVIQ':
            path = '/home1/mpc/python_projects/OIQA_KD/cross_database/model_cmp_base_OIQA_CVIQ_epoch3_0.8771_0.9056_6.0605.pkl'
        elif database_test == 'IQA-ODI':
            path = '/home1/mpc/python_projects/OIQA_KD/cross_database/model_cmp_base_OIQA_IQA-ODI_epoch18_-0.6654_0.6567_15.1219.pkl'
    elif database_train == 'IQA-ODI':
        if database_test == 'CVIQ':
            path = '/home1/mpc/python_projects/OIQA_KD/cross_database/model_cmp_base_IQA-ODI_CVIQ_epoch34_-0.7974_0.8343_7.8772.pkl'
        elif database_test == 'OIQA':
            path = '/home1/mpc/python_projects/OIQA_KD/cross_database/model_cmp_base_IQA-ODI_OIQA_epoch2_-0.5038_0.5636_11.8835.pkl'

    teacher.load_state_dict(torch.load(path))
    teacher.eval()
    
    # ---image item list for IQA-ODI database---
    # ID_list = []
    # ID_path = '/home1/mpc/Dataset/IQA-ODI/RefImp_ID.txt'
    # root = '/home1/mpc/Dataset/IQA-ODI/resized_cubic/'
    # with open(ID_path, 'r') as f:
    #     for line in f:
    #         ID_list.append(line.rstrip('\n'))
    
    root_path = './feature_all/'
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    save_path = './feature_all/' + database_train + '-' + database_test
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # CVIQ: range(544)
    # OIQA: range(320)
    # IQA-ODI: range(1080)
    for i in range(320):
        # ---image item list for IQA-ODI database---
        # img_index = ID_list[i]
        # img_path = root + img_index
        img_index = str(i + 1).rjust(3, '0')
        img_path = '/home1/mpc/Dataset/' + database_train + '/resized_cubic/' + img_index

        img_path1 = img_path + 'F.png'
        img_path2 = img_path + 'R.png'
        img_path3 = img_path + 'BA.png'
        img_path4 = img_path + 'L.png'
        img_path5 = img_path + 'T.png'
        img_path6 = img_path + 'BO.png'

        img1 = Image.open(img_path1)
        img1 = img1.convert('RGB')
        img1 = img1.resize((256, 256))
        img1 = transforms.ToTensor()(img1).unsqueeze(0).cuda()

        img2 = Image.open(img_path2)
        img2 = img2.convert('RGB')
        img2 = img2.resize((256, 256))
        img2 = transforms.ToTensor()(img2).unsqueeze(0).cuda()

        img3 = Image.open(img_path3)
        img3 = img3.convert('RGB')
        img3 = img3.resize((256, 256))
        img3 = transforms.ToTensor()(img3).unsqueeze(0).cuda()

        img4 = Image.open(img_path4)
        img4 = img4.convert('RGB')
        img4 = img4.resize((256, 256))
        img4 = transforms.ToTensor()(img4).unsqueeze(0).cuda()

        img5 = Image.open(img_path5)
        img5 = img5.convert('RGB')
        img5 = img5.resize((256, 256))
        img5 = transforms.ToTensor()(img5).unsqueeze(0).cuda()

        img6 = Image.open(img_path6)
        img6 = img6.convert('RGB')
        img6 = img6.resize((256, 256))
        img6 = transforms.ToTensor()(img6).unsqueeze(0).cuda()

        _, feat = teacher(img1, img2, img3, img4, img5, img6)
        # feat = feat.cpu().detach()
        save_path = './feature_all/' + database_train + '-' + database_test + '/' + img_index + '.npy'
        torch.save(feat, save_path)
