import torch
from torch import nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    
    """
        Parameter :
        dim -> basically input's shape
        expension_factor -> the number of neurons in the intermediate (hidden) layer. 
        dropout -> regularization to prevent overfitting and imporve generalization ability of model 
        
    """    
    def __init__(self, dim,expension_factor,dropout):
        super().__init__()
        hidden_dim = dim * expension_factor
        self.fc1 = nn.Linear(dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,dim)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
    
    def forward(self,x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        return self.dropout2(self.fc2(x))
        
        
        
class Fourier(nn.Module):
    def __init__(self,input):
        super().__init__()
        self.dropuout = nn.Dropout(0.3)
        self.act = nn.ReLU(input)
        
    def forward(self,x):
        x= torch.fft.fft(x,dim=-1)
        x = torch.fft.fft(x,dim = 1)
        x = self.act(x)
        x = self.dropout(x)
        return x 


class Fnet(nn.Module):
    def __init__(self,dim,expension_factor, dropout):
        super().__init__()
        self.fourier = Fourier()
        self.ffn = FeedForward(dim,expension_factor,dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def __call__(self,x):
        residual = x
        x = self.fourier(x)
        x = self.norm1(x+residual)
        residual = x
        x = self.ffn(x)
        out = self.norm2(x+residual)
        return out 
    