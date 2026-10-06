import torch
import torch.nn as nn

class ECG_RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, num_classes=5, dropout=0.5, model_type='lstm'):
        super(ECG_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type.lower()

        if self.model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout, bidirectional=True)
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout, bidirectional=True)
        else:
            raise ValueError("Chỉ hỗ trợ 'lstm' hoặc 'gru'")

        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out, _ = torch.max(rnn_out, dim=1) 
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out