import torch
from torch import nn
from torch.nn import functional as F
import jax
import jax.numpy as jnp

class FeedForward(nn.Module):
    def __init__(self, num_features,expansion_factor,dropout):
        super().__init__()
        hidden_dim = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,num_features)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
    
    def forward(self,x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        return self.dropout2(self.fc2(x))
        
        
        
        