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
import dataset_kd as dataset
import time
import random
from tqdm import tqdm
import resnet_erp as model_erp

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
    parser.add_argument('--path', dest='path', help='Root path of the database', type=str, default='./data/')
    parser.add_argument("--T", type=float, default=4.0)
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':

    args = parse_args()
    s_t = time.time()
    device = torch.device('cuda:0')
    set_seed(2023)
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    snapshot = args.snapshot
    database = args.database
    db_path = args.path + args.database
    lr = args.lr
    
    # writer = SummaryWriter('./logs')

    print('————————————————————————————————————————')
    print('The training set of Mutual KD:')
    print('The train epoch  : {}'.format(num_epochs))
    print('The batch size   : {}'.format(batch_size))
    print('The learning rate: {}'.format(lr))
    print('Training dataset : {}'.format(database))   
    print('————————————————————————————————————————')
    
    filename_train = db_path + '/' + database + '_trainc_'
    filename_test = db_path + '/' + database + '_testc_'

    totle = [0, 0, 0, 0]
    # 5-fold cross-validation
    for loop in range(5):
        
        student = model_erp.resnet18(pretrained=True).cuda()
        p_net = policy_Net().cuda()
        optimizer = torch.optim.Adam([{'params': student.parameters()},
                                     {'params': p_net.parameters(), 'lr': lr/2}], lr=lr, weight_decay=5e-4)
        
        criterion = nn.MSELoss()   
        criterion_kd = nn.MSELoss()
        
        file_train = filename_train + str(loop + 1) + '.csv'
        file_test = filename_test + str(loop + 1) + '.csv'

        train_dataset = dataset.Dataset(file_train, database, loop)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                   worker_init_fn=seed_worker)
        test_dataset = dataset.Dataset(file_test, database, loop)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        best_val_criterion = -1
        best_val = []
        
        for epoch in range(num_epochs):
            student.train()
            p_net.train()

            pbar = tqdm(train_loader)
            for i, (img, feat_t, mos) in enumerate(pbar):
                img = img.cuda()
                feat_t = feat_t.cuda()
                mos = mos[:, np.newaxis]
                mos = mos.cuda()

                score, feat = student(img)

                loss_sup = criterion(score, mos)

                if epoch < 5:
                    loss = loss_sup
                    pbar.set_description('The %d th loop-Epoch [%d/%d], loss_erp: %4f' % 
                                         (loop + 1, epoch + 1, num_epochs, loss_sup.data.item()))

                elif 5 <= epoch < 10:
                    loss_kd = criterion_kd(feat, feat_t)
                    loss = loss_sup + 0.2 * loss_kd
                    pbar.set_description('The %d th loop-Epoch [%d/%d], loss_erp: %4f, loss_kd: %4f' % 
                                         (loop + 1, epoch + 1, num_epochs, loss_sup.data.item(), 0.2 * loss_kd.data.item()))

                else:
                    mask, f_e_m = p_net(feat.clone().detach(), feat_t)
                    s_e = student.fc1(f_e_m)
                    loss_p = criterion(s_e, mos)
                    loss_kd = criterion_kd(feat, f_e_m.clone().detach())
                    loss = loss_sup + loss_p + 0.2 * loss_kd

                    pbar.set_description(
                        'The %d th loop-Epoch [%d/%d], loss_erp: %4f, loss_kd: %4f, loss_p: %4f'
                        % (loop + 1, epoch + 1, num_epochs, loss_sup.data.item(), 0.2 * loss_kd.data.item(), loss_p.data.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            with torch.no_grad():
                student.eval()
                p_net.eval()

                label = np.zeros([len(test_dataset)])
                y_output = np.zeros([len(test_dataset)])
                
                for i, (img, _, mos) in enumerate(test_loader):
                    img = img.cuda()

                    mos = mos[:, np.newaxis]
                    mos = mos.cuda()

                    label[i] = mos

                    score, feat = student(img)
                    
                    y_output[i] = score.item()

                label = np.array(label)
                y_output = np.array(y_output)
                y_output_logistic = fit_function(label, y_output)

                val_PLCC = stats.pearsonr(y_output_logistic, label)[0]
                val_SRCC = stats.spearmanr(y_output, label)[0]
                val_KRCC = stats.stats.kendalltau(y_output, label)[0]
                val_RMSE = np.sqrt(((y_output_logistic - label) ** 2).mean())

                print('Epoch {} on ERP completed. SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(epoch + 1, \
                                                                                                              val_SRCC,
                                                                                                              val_KRCC,
                                                                                                              val_PLCC,
                                                                                                              val_RMSE))

                if val_SRCC > best_val_criterion:
                    print("—————— Update best ERP model using best_val_criterion in epoch {}——————".format(epoch + 1))
                    best_val_criterion = val_SRCC
                    best_val = [val_SRCC, val_KRCC, val_PLCC, val_RMSE]
                    # save the model file
                    # if not os.path.exists('./model'):
                    #     os.mkdir('./model')
                    # if not os.path.exists('./model/' + database):
                    #     os.mkdir('./model/' + database)
                    # torch.save(student.state_dict(), os.path.join('./model/' + database + '/' + snapshot + '_erp_' + database + '_loop' + str(loop+1) + '_epoch' + str(epoch+1) + '_' + str('{:.4f}_{:.4f}_{:.4f}'.format(val_SRCC, val_PLCC, val_RMSE)) + '.pkl'))

            print('Knowledge distillation Training on {} completed.'.format(database))
            print('The best training result of ERP model SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_val[0], best_val[1], best_val[2], best_val[3]))

        totle[0] = totle[0] + best_val[0]
        totle[1] = totle[1] + best_val[1]
        totle[2] = totle[2] + best_val[2]
        totle[3] = totle[3] + best_val[3]

    e_t = time.time()
    print('FINAL —— The result on {} of Mutual model:'.format(database))
    print('————————————————————————————————————————')
    print('The training set of Mutual KD:')
    print('The train epoch  : {}'.format(num_epochs))
    print('The batch size   : {}'.format(batch_size))
    print('The learning rate: {}'.format(lr))
    print('Training dataset : {}'.format(database))
    print('————————————————————————————————————————')
    # the average result of 5-fold cross-validation
    print('ERP model: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
        totle[0] / 5, totle[1] / 5, totle[2] / 5, totle[3] / 5))
    print('The total time used is :{:.4f}'.format(e_t - s_t))
    