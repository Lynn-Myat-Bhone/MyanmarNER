import unittest
import torch
from embedding import Embedding ,PositionalEncoding

class TestEmbedding(unittest.TestCase):
    def test_embedding_and_positional_encoding(self):
        vocab_size = 33
        embedding_dim = 30
        batch_size = 8
        seq_len = 40
        
        embedding = Embedding(vocab_size,embedding_dim)
        pos_en = PositionalEncoding(seq_len,embedding_dim)
        input_ids = torch.randint(0,vocab_size,(batch_size,seq_len))
        embedded = embedding(input_ids)
        pos_encoded = pos_en(embedded)
        
        self.assertEqual(pos_encoded.shape,(batch_size,seq_len,embedding_dim))
        self.assertFalse(torch.equal(pos_encoded, embedded))
        
