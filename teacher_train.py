import os, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
import resnet_cmp
import dataset_viewport
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import random

from scipy import stats
from scipy.optimize import curve_fit


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
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
    parser.add_argument('--database', dest='database', help='The database that needs to be trained and tested.',
                        default='OIQA', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=6, type=int)
    parser.add_argument('--cross_validation_index', dest='cross_validation_index',
                        help='The index of cross validation.',
                        default='1', type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
                        default='baseline_resnet18', type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    
    args = parse_args()
    totle = [0, 0, 0, 0]
    set_seed(2023)
    s_t = time.time()
    
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    snapshot = args.snapshot
    database = args.database
    lr = args.lr
    device = torch.device('cuda:0') 
    
    if database == 'CVIQ':
        filename_train = './data/CVIQ/CVIQ_trainc_'
        filename_test = './data/CVIQ/CVIQ_testc_'
    elif database == 'OIQA':
        filename_train = './data/OIQA/OIQA_trainc_'
        filename_test = './data/OIQA/OIQA_testc_'
    elif database == 'IQA-ODI':
        filename_train = './data/IQA-ODI/ODI_trainc_'
        filename_test = './data/IQA-ODI/ODI_testc_'

    for loop in range(5):   

        model = resnet_cmp.resnet18(pretrained=True).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=5e-4)      
        
        file_train = filename_train + str(loop+1) + '.csv'
        file_test = filename_test + str(loop+1) + '.csv'     
        
        train_dataset = dataset_viewport.Dataset(database, file_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        test_dataset = dataset_viewport.Dataset(database, file_test)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)             

        best_val_criterion = -1
        best_val = []
        for epoch in range(num_epochs):
            model.train()

            pbar = tqdm(train_loader)
            for i, (img1, img2, img3, img4, img5, img6, mos) in enumerate(pbar):
                img1 = img1.to(device)
                img2 = img2.to(device)
                img3 = img3.to(device)
                img4 = img4.to(device)
                img5 = img5.to(device)
                img6 = img6.to(device)

                mos = mos[:, np.newaxis]
                mos = mos.to(device)

                optimizer.zero_grad()

                mos_predict, _ = model(img1, img2, img3, img4, img5, img6)
                loss = criterion(mos_predict, mos)

                loss.backward()
                optimizer.step()
                
                pbar.set_description(
                    'The %d th loop-Epoch [%d/%d], loss: %4f'
                    % (loop + 1, epoch + 1, num_epochs, loss.data.item()))

            with torch.no_grad():
                model.eval()
                label = np.zeros([len(test_dataset)])
                y_output = np.zeros([len(test_dataset)])
                for i, (img1, img2, img3, img4, img5, img6, mos) in enumerate(test_loader):
                    img1 = img1.to(device)
                    img2 = img2.to(device)
                    img3 = img3.to(device)
                    img4 = img4.to(device)
                    img5 = img5.to(device)
                    img6 = img6.to(device)
                
                    mos = mos[:, np.newaxis]
                    mos = mos.to(device)

                    label[i] = mos

                    mos_predict, _ = model(img1, img2, img3, img4, img5, img6)
                    y_output[i] = mos_predict.item()

                label = np.array(label)
                y_output = np.array(y_output)
                y_output_logistic = fit_function(label, y_output)
            
                val_PLCC = stats.pearsonr(y_output_logistic, label)[0]
                val_SRCC = stats.spearmanr(y_output, label)[0]
                val_KRCC = stats.stats.kendalltau(y_output, label)[0]
                val_RMSE = np.sqrt(((y_output_logistic - label) ** 2).mean())

                print('Epoch {} completed. SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(epoch + 1, \
                                                                                                              val_SRCC,
                                                                                                              val_KRCC,
                                                                                                              val_PLCC,
                                                                                                              val_RMSE))

                if val_SRCC + val_PLCC > best_val_criterion:
                    print("Update best model using best_val_criterion in epoch {}".format(epoch + 1))
                    best_val_criterion = val_SRCC + val_PLCC
                    best_val = [val_SRCC, val_KRCC, val_PLCC, val_RMSE]
                    
                    if not os.path.exists('./baseline')
                        os.mkdir('./baseline')
                    torch.save(model.state_dict(), os.path.join('./baseline/' + snapshot + '_' + database + '_' + str(loop+1) + '_epoch' + str(epoch+1)
                                                                 + '_' + str('{:.4f}_{:.4f}_{:.4f}'.format(val_SRCC, val_PLCC, val_RMSE)) + '.pkl'))
                    # torch.save(model.state_dict(), os.path.join(snapshot + '_viewport_' + database + '_epoch' + str(epoch+1)
                    #                                             + '_' + str('{:.4f}_{:.4f}_{:.4f}'.format(val_SRCC, val_PLCC, val_RMSE)) + '.pkl'))

            print('Viewport model Training on {} completed.'.format(database))
            print('The best training result SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_val[0], best_val[1], best_val[2], best_val[3]))
        totle[0] = totle[0] + best_val[0]    
        totle[1] = totle[1] + best_val[1] 
        totle[2] = totle[2] + best_val[2] 
        totle[3] = totle[3] + best_val[3]        
    e_t = time.time()
    print('The result on {} of viewport_model:'.format(database))
    print('SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            totle[0]/5, totle[1]/5, totle[2]/5, totle[3]/5))
    print('The total time used is :{:.4f}'.format(e_t-s_t))