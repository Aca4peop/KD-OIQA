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
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
                        default=100, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate.',
                        default=0.0002, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=6, type=int)
    parser.add_argument('--database', dest='database', help='The database that needs to be trained and tested.',
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

    database = args.database

    for loop in range(5):
        teacher = model_cmp.resnet18().cuda()
        if loop == 0:
            if database == 'CVIQ':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_CVIQ_1_epoch41_0.9553_0.9584_3.7796.pkl'
            elif database == 'OIQA':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_OIQA_1_epoch37_0.9586_0.9581_4.1400.pkl'
            elif database == 'IQA-ODI':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_IQA-ODI_1_epoch38_0.9232_0.9260_7.7663.pkl'
                
        elif loop == 1:
            if database == 'CVIQ':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_CVIQ_2_epoch20_0.9434_0.9573_3.9559.pkl'
            elif database == 'OIQA':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_OIQA_2_epoch13_0.8655_0.8858_6.9313.pkl'
            elif database == 'IQA-ODI':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_IQA-ODI_2_epoch22_0.8939_0.8815_9.4126.pkl'

        elif loop == 2:
            if database == 'CVIQ':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_CVIQ_3_epoch35_0.9455_0.9511_4.3332.pkl'
            elif database == 'OIQA':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_OIQA_3_epoch11_0.9420_0.9457_4.9879.pkl'
            elif database == 'IQA-ODI':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_IQA-ODI_3_epoch16_0.8739_0.8594_10.1370.pkl'

        elif loop == 3:
            if database == 'CVIQ':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_CVIQ_4_epoch41_0.9188_0.9561_4.3055.pkl'
            elif database == 'OIQA':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_OIQA_4_epoch16_0.9436_0.9495_4.7419.pkl'
            elif database == 'IQA-ODI':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_IQA-ODI_4_epoch16_0.9211_0.8705_9.7351.pkl'

        else:
            if database == 'CVIQ':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_CVIQ_5_epoch72_0.9481_0.9568_4.1837.pkl'
            elif database == 'OIQA':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_OIQA_5_epoch34_0.9664_0.9695_3.5721.pkl'
            elif database == 'IQA-ODI':
                path = '/home1/mpc/OIQA_KD/baseline/model_cmp_base_IQA-ODI_5_epoch31_0.9264_0.8806_9.5090.pkl'

        teacher.load_state_dict(torch.load(path))
        teacher.eval()
        
        # ---image item list for IQA-ODI database---
        # ID_list = []
        # ID_path = '/home1/mpc/Dataset/IQA-ODI/RefImp_ID.txt'
        # root = '/home1/mpc/Dataset/IQA-ODI/resized_cubic/'
        # with open(ID_path, 'r') as f:
        #     for line in f:
        #         ID_list.append(line.rstrip('\n'))
        
        # CVIQ: range(544)
        # OIQA: range(320)
        # IQA-ODI: range(1080)
        for i in range(320):
            # ---img_index for IQA-ODI database---
            # img_index = ID_list[i]
            # img_path = root + img_index
            img_index = str(i + 1).rjust(3, '0')
            img_path = '/home1/mpc/Dataset/' + database + '/resized_cubic/' + img_index

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
            save_path = '/home1/mpc/OIQA_KD/features/' + database + '/' + str(loop+1) + '/' + img_index + '.npy'
            torch.save(feat, save_path)
