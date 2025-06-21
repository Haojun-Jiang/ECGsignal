import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np

class cnn_1d(nn.Module):
    def __init__(self, classes_n = 4):
        super(cnn_1d, self).__init__() # input 1 x 5000
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # => 32 × 2500
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # => 64 × 1250
            nn.Dropout(0.1),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # => 128 × 625
            nn.Dropout(0.1),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # => 256 × 312
            nn.Dropout(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 312, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, classes_n)
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x
    
class bi_lstm(nn.Module):
    def __init__(self, input_size=256, hidden_size=100, num_layers=2, output_size=4):
        super(bi_lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # 2×hidden_size due to bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.classifier(out)   #[B, output_size]
        return out

class cnn_feed_lstm(nn.Module):
    def __init__(self, cnn_model, rnn_model, classes_n = 4):
        super(cnn_feed_lstm, self).__init__()
        self.cnn = cnn_model
        self.lstm = rnn_model
    
    def forward(self, x):
        feature = self.cnn(x)
        feature = feature.permute(0,2,1)
        out = self.lstm(feature)
        return out

