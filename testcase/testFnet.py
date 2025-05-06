import torch
from torch import nn
from torch.nn import functional as F
import unittest
from Engine.fNet import FNet 


input_dim = 128
batch_size = 32  
expansion_factor  = 4
dropout = 0.3
x = torch.randn(batch_size,input_dim) # (32,128)

class TestFnet(unittest.TestCase):
    def testFnet(self):
        model = FNet(input_dim,expansion_factor,dropout)
        out = model(x)
        
        # Check input and output shape are consistent throughout layer
        assert out.shape == x.shape,f"Expected output shape {x.shape}, but got {out.shape}"
        
        # Ensure model can successfully run a forward pass without error
        assert out is not None, "Model forward pass failed, output is None."
        
         # Check for NaN or Inf in the output
        assert not torch.any(torch.isnan(out)), "Output contains NaN values."
        assert not torch.any(torch.isinf(out)), "Output contains Inf values."
        
        # Check Gradient Flow During Backpropagation
        loss = out.sum()
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None, f"Gradients for parameter {param} are None."
            assert not torch.all(param.grad == 0), f"Gradients for parameter {param} are all zero."

        
    