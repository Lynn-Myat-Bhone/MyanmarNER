import torch
from torch import nn
import math

class Embedding(nn.Module):
    def __init__(self,vocab_size,embedding_dim):
        super().__init__()
        self.weight =nn.Parameter( torch.randn(vocab_size,embedding_dim)) #this is learnable paramter
        print(self.weight.shape)
        
    def forward(self,input):
        return self.weight[input]
        
class PositionalEncoding(nn.Module):
    def __init__(self,max_len,d_model):
        super().__init__()
        pe = torch.zeros(max_len,d_model) 
        position = torch.arange(max_len).unsqueeze(1) # shape[max_len,1]
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)  # Register the positional encoding tensor as a buffer
        
    def forward(self,x):
        # x (batch_size,max_len, d_model)
        x = x + self.pe[:,:x.size(1)] #BroadCast positional encoding
        return x