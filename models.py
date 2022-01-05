import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.datasets import imdb
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUnet(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=500, hidden_dim=500, layer_dim=1, output_dim=2, bidirection=False, pretrained_embd=''):
        super(GRUnet, self).__init__()
        self.model_type = 'GRU'
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.bidirection = bidirection

        if len(pretrained_embd) > 1:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(get_embed_vec(vocab_size, pretrained_embd))).to(device)
            self.embedding.requires_grad_(False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(embedding_dim, hidden_dim, layer_dim, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            torch.nn.Mish(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        scaler = 2 if self.bidirection else 1
        h_0 = torch.zeros(self.layer_dim * scaler, x.size(0), self.hidden_dim).to(device)

        embeds = self.embedding(x)
        r_out, h_n = self.gru(embeds, h_0)
        out = self.decoder(r_out[:, -1, :])
        # out : [batch, time_step, output_dim]
        return out 


class LSTMnet(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=500, hidden_dim=1000, layer_dim=1, output_dim=2, bidirection=False, pretrained_embd=''):
        super(LSTMnet, self).__init__()
        self.model_type = 'LSTM'
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.bidirection = bidirection

        if len(pretrained_embd) > 1:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(get_embed_vec(vocab_size, pretrained_embd))).to(device)
            self.embedding.requires_grad_(False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim,
                         batch_first=True,bidirectional=self.bidirection)
       
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            torch.nn.Mish(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            #nn.Dropout(0.1),
            torch.nn.Mish(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        scaler = 2 if self.bidirection else 1
        h_0 = torch.zeros(self.layer_dim * scaler, x.size(0), self.hidden_dim).to(device)
        c_0 = torch.zeros(self.layer_dim * scaler, x.size(0), self.hidden_dim).to(device)

        embeds = self.embedding(x)

        r_out, (h_n, c_n) = self.lstm(embeds, (h_0, c_0))

        out = self.decoder(r_out[:, -1, :])

        return out 


def get_embed_vec(vocab_size, embedding_path):

    word2idx = imdb.get_word_index()
    # inverted_word_index = dict((i, word) for (word, i) in word2idx.items())
    tuple1=zip(word2idx.values(),word2idx.keys())
    idx_word=list(sorted(tuple1))
    
    embeddings = {}
    with open(embedding_path, encoding='utf-8') as gf:
        for glove in gf:
            word, embedding = glove.split(maxsplit=1)
            embedding = [float(s) for s in embedding.split(' ')]
            embeddings[word] = embedding
    
    dim = len(embedding)
    vocab_embeddings = np.zeros((vocab_size, dim))
    vocab_embeddings[:3] = torch.rand((3,dim))
    for idx, word in idx_word[:vocab_size - 3]:
        try:
            embedding = embeddings[word]
        except:
            embedding = torch.rand((1, dim))
        vocab_embeddings[idx + 2, :] = embedding

    return vocab_embeddings



class TransformerModel(nn.Module):

    def __init__(self, squ_len: int, vocab_size: int, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.squ_len = squ_len
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src_mask = generate_square_subsequent_mask(self.squ_len)

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask)
        output = output.permute(0, 2, 1)                #[batch_size squence_len d_model] -> [batch_size d_model squence_len]

        output = self.global_avg_pool(output)
        output = torch.squeeze(output)                  #[batch_size d_model 1] -> [batch_size d_model]

        output = self.decoder(output)
        
        return output


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).to(device)                #leave room for dim-batch_size
        # self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]                      #positional ecoding by summation
        return self.dropout(x)
