import sys
import os
import torch
from torch import nn
import torch.nn.functional as F

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from BaseLanguageModel import BaseLanguageModel

class GRULanguageModel(BaseLanguageModel):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=6, dropout=0.2, pad_token_id=3):
        super().__init__(vocab_size, embed_dim, pad_token_id, model_name="GRU")
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        """
        Compute model output logits given a sequence
        :param input_ids: Sequence of input token IDs, Tensor of shape (batch_size, seq_len)
        :return: output logits, Tensor of shape (batch_size, seq_len, vocab_size)
        """
        embeds = self.embedding(input_ids)
        output, hidden = self.gru(embeds, hidden)
        return self.fc(output), hidden
