import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import LSTMnet,GRUnet,TransformerModel


def select_device(device : str=''):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        import torch
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    return torch.device('cuda:0' if cuda else 'cpu')


def get_hyp(hyp_path: str):
    curPath = os.path.dirname(os.path.realpath(__file__))
    yamlPath = os.path.join(curPath, hyp_path)
    f = open(yamlPath, 'r', encoding='utf-8')
    hyp = yaml.load(f.read(), Loader=yaml.FullLoader)

    return hyp


def select_model(opt, hyp :dict):
    assert opt.model.lower() in ['gru', 'lstm', 'transformer'], 'choose one of GRU, LSTM and Transformer'
    if opt.model.lower() == 'gru':
        return GRUnet(pretrained_embd=opt.pretrained)
    elif opt.model.lower() == 'lstm':
        return LSTMnet(pretrained_embd=opt.pretrained)
    else:
        return TransformerModel(squ_len = hyp['sequence_padding'], vocab_size = 10000, ntoken=2, 
                                d_model=128, nhead=4, d_hid=128, nlayers=2, dropout=0.5)


def write2log(log_dict, lr, train_loss, test_loss, test_acc, isLastepoch, exp_path):
    
    log_dict['learning_rate'].append(round(lr, 6))
    log_dict['train_loss'].append(round(train_loss,4))
    log_dict['test_loss'].append(round(test_loss,4))
    log_dict['test_acc'].append(round(test_acc,3))
    if isLastepoch:
        log_pd = pd.DataFrame(log_dict)
        sns.set_theme(style='whitegrid',font_scale=1.2)
        plt.figure(1, figsize=(9,6))
        p = sns.lineplot(data = log_pd[['train_loss', 'test_loss']])
        p.set_xlabel('epoch')
        p.set_ylabel('loss')
        p.set_title('Train & Test Loss over Epochs')
        plt.savefig(exp_path + '/loss.png')
        plt.figure(2, figsize=(9,6))
        p_acc = sns.lineplot(data = log_pd['test_acc'])
        p_acc.set_xlabel('epoch')
        p_acc.set_title('Accuracy over Epochs')
        plt.savefig(exp_path + '/acc.png')
