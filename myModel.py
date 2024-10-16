import torch
import torch.nn as nn

class TranscriptionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TranscriptionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)


    
