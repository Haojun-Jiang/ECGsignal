from data import ECGDataset, ECGsegments, STFTgraph
from helper_code import *
from models import *
import helper_code as hc
import torch
import os
from train_module import *
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

def train_2d_STFT():
    cnn = CNN2D_3layers()
    rnn = bi_lstm(input_size = 64*30)
    cnn_feed_rnn = CNN2D_feed_lstm(cnn, rnn)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_feed_rnn.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    raw_data = STFTgraph(
        root_dir = './data/spectrogram',
        transform = transform
    )
    train_dataset, val_dataset = random_split(raw_data, [int(len(raw_data)*0.8), len(raw_data)-int(len(raw_data)*0.8)])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)

    results = train_module(cnn_feed_rnn, train_loader, val_loader, optimizer, criterion, device='cuda',n_epochs=50)
    return results
    return 0

results = train_2d_STFT()
os.makedirs('./traininglog', exist_ok=True)
torch.save(results, f"traininglog/train_2d_3layers.pt")