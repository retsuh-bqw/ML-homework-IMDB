import os
import math
from numpy.lib.function_base import select
import yaml
import torch
import argparse
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from dataset import create_dataloader
from utills import select_model, select_device, get_hyp, write2log



def train(hyp,model,device,train_loader,test_loader):
    # create a folder for saving exp info
    exp_path = './exp_'+datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '_' + model.model_type
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    log = {'learning_rate': [], 'train_loss':[], 'test_loss':[],'test_acc':[]}

    # optimizer setting
    if hyp['optim'] == 'Adam':
        optimizer = optim.Adam( model.parameters(), lr=hyp['lr'],
                    weight_decay=hyp['weight_decay'], betas=(hyp['momentum'],0.999))
    else:
        optimizer = optim.SGD(model.parameters(), lr=hyp['lr'], 
                    weight_decay=hyp['weight_decay'], momentum=hyp['momentum'])

    # cosine annealing
    epochs = hyp['eps'] 
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.2) + 0.2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Softmax + BCE
    loss_func = nn.CrossEntropyLoss() 

    best_acc = 0.0
    losses = []
    for epoch in range(1, epochs+1): 
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('Currnet lr: ',cur_lr)
        
        model.train()

        nb = len(train_loader)          #number of batches
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=nb)

        for step,(x,y) in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            y_ = model(x)	
            loss = loss_func(y_,y)
            losses.append(loss.item())
            loss.backward()	
            optimizer.step()
            
            s = 'Train Epoch:{} [{}/{} ({:.0f}%)]  Loss:{:.6f}'.format(
                    epoch, step*len(x),len(train_loader.dataset),
                    100.*step/len(train_loader),loss.item())
            pbar.set_description(s)

        scheduler.step()
        test_loss, cur_acc = test(model, device, test_loader)
        if cur_acc > best_acc:
            best_acc = cur_acc
            torch.save(model, exp_path + hyp['save_path'])
        print("current acc is:{:.4f},best acc is {:.4f}\n".format(cur_acc,best_acc))
        
        write2log(log, cur_lr, np.mean(losses), test_loss.item(), cur_acc, True if epoch == epochs else False, exp_path)
        losses.clear()
    
    # saving log file
    pd.DataFrame(log).to_csv(exp_path + '/train_log.csv',index=False, sep=',')
    with open(exp_path + '/hyp.yaml',"w", encoding="utf-8") as f:
        yaml.dump(hyp, f)



def test(model,device,test_loader):
    model.eval()

    loss_func = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0.0
    acc = 0
    for step,(x,y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            y_ = model(x)

        test_loss += loss_func(y_,y)
        pred = y_.max(-1,keepdim=True)[1]
        acc += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set:Average loss:{:.4f},Accuracy:{}/{} ({:.0f}%)'.format(
        test_loss,acc,len(test_loader.dataset),
        100*acc/len(test_loader.dataset)
    ))

    return test_loss, acc / len(test_loader.dataset)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Transformer', help='model selected to be trained')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--hyp', type=str, default='hyp.yaml', help='hyperparameters cfg pth')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained embedding path')
    opt = parser.parse_args()

    # get hyperparameters
    hyp = get_hyp(opt.hyp)

    # select device
    device = select_device(opt.device)

    # define model
    model = select_model(opt, hyp).to(device)

    # create dataset
    train_loader, test_loader = create_dataloader(batch_size=hyp['bs'], max_len=hyp['sequence_padding'])

    print('Start training...')
    train(hyp, model, device, train_loader, test_loader)


