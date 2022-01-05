import numpy as np
import torch
from torch.utils.data import RandomSampler,DataLoader,TensorDataset,SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb


def create_dataloader(batch_size, max_len=500):
    
    (x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=10000)

    x_train = pad_sequences(x_train,maxlen=max_len,padding='post',truncating='post')
    x_test = pad_sequences(x_test,maxlen=max_len,padding='post',truncating='post')

    # to Tensor
    train_data = TensorDataset(torch.LongTensor(x_train),torch.LongTensor(y_train))
    test_data = TensorDataset(torch.LongTensor(x_test),torch.LongTensor(y_test))

    # create dataloader
    bs = batch_size
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data,sampler=train_sampler,batch_size=bs)
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data,sampler=test_sampler,batch_size=bs)

    return train_loader, test_loader