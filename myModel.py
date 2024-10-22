import torch
import torch.nn as nn

class TranscriptionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob = 0.3):
        super(TranscriptionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        return self.fc(lstm_out)