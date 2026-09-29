"""
Neural network architectures for Isolated Sign Language Recognition.
"""

import torch
import torch.nn as nn


class SLRModel(nn.Module):
    """
    Baseline model for Isolated SLR on landmarks (e.g., BiLSTM / GRU / Transformer).
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super(SLRModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Take the output at the last time step or mean pooling
        out = torch.mean(out, dim=1)
        out = self.fc(out)
        return out
