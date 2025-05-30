import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from TorchCRF import CRF

class TransformerNER(nn.Module):
    def __init__(self, embedding_matrix, num_ner_tags, num_pos_tags, dropout=0.3, num_layers=2):
        super(TransformerNER, self).__init__()
        dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.embedding_dropout = nn.Dropout(dropout)

        self.position_embedding = nn.Embedding(512, dim)  # max_len = 512

        encoder_layer = TransformerEncoderLayer(
            d_model=dim,
            nhead=6,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ner_classifier = nn.Linear(dim, num_ner_tags)
        self.pos_classifier = nn.Linear(dim, num_pos_tags)
        self.crf_ner = CRF(num_ner_tags, batch_first=True)
        

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.position_embedding(positions)

        # PyTorch expects padding mask as True for pad
        x = self.encoder(x, src_key_padding_mask=~mask)

        ner_logits = self.ner_classifier(x)
        pos_logits = self.pos_classifier(x)
        return ner_logits, pos_logits

    def loss(self, ner_emissions, ner_tags, pos_logits, pos_tags, mask, alpha=1.0, ner_weights=None, pos_weights=None):
        # NER loss (weighted token-level CrossEntropy if weights provided, otherwise CRF)
        ner_logits_flat = ner_emissions.view(-1, ner_emissions.size(-1))
        ner_targets_flat = ner_tags.view(-1)

        if ner_weights is not None:
            token_loss = F.cross_entropy(ner_logits_flat, ner_targets_flat, weight=ner_weights, ignore_index=0)
        else:
            token_loss = -self.crf_ner(ner_emissions, ner_tags, mask=mask, reduction='mean')

        # POS loss (always CrossEntropy)
        pos_logits_flat = pos_logits.view(-1, pos_logits.size(-1))
        pos_targets_flat = pos_tags.view(-1)

        ce_loss = F.cross_entropy(pos_logits_flat, pos_targets_flat, weight=pos_weights, ignore_index=0)

        return token_loss + alpha * ce_loss, token_loss.item(), ce_loss.item()

    def decode(self, ner_emissions, mask):
        return self.crf_ner.decode(ner_emissions, mask=mask)

