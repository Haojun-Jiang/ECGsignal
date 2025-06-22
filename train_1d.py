from data import ECGDataset, ECGsegments
from helper_code import *
from models import *
import helper_code as hc
import torch
import os
from train_module import *
from torch.utils.data import Dataset, DataLoader, random_split

def load_label_map(txt_path):
    with open(txt_path) as f:
        codes = [line.strip() for line in f if line.strip()]
    return {code: i for i, code in enumerate(codes)}

def train_1d():
    cnn = cnn_1d()
    rnn = bi_lstm()
    cnn_feed_rnn = cnn_feed_lstm(cnn, rnn)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_feed_rnn.parameters(), lr=0.001)

    label_map = load_label_map('D:\study\Msc project\project\scr\classes.txt')
    raw_data = ECGDataset(
    data_dirs=['data/training/chapman_shaoxing','data/training/ningbo'],
    label_map=label_map,
    signal_length=5000,
    leads=['II']
    )

    train_dataset, val_dataset = random_split(raw_data, [int(len(raw_data)*0.8), len(raw_data)-int(len(raw_data)*0.8)])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

    results = train_module(cnn_feed_rnn, train_loader, val_loader, optimizer, criterion, device='cuda',n_epochs=50)
    return results

# results=train_1d()
# os.makedirs('./traininglog', exist_ok=True)
# torch.save(results, f"./traininglog/train_1d_v1.pt")

def train_1d_segments():
    cnn = cnn_1d()
    rnn = bi_lstm()
    cnn_feed_rnn = cnn_feed_lstm(cnn, rnn)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_feed_rnn.parameters(), lr=0.001)

    try:
        segments = np.load('./data/segments.npy')
    except:
        segments = hc.segments(['data/training/chapman_shaoxing','data/training/ningbo'])

    raw_data = ECGsegments('data/segments.npy', segments=segments, segment_length=1000)

    train_dataset, val_dataset = random_split(raw_data, [int(len(raw_data)*0.8), len(raw_data)-int(len(raw_data)*0.8)])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

    results = train_module(cnn_feed_rnn, train_loader, val_loader, optimizer, criterion, device='cuda',n_epochs=2)
    return results

results = train_1d_segments()
os.makedirs('./traininglog', exist_ok=True)
torch.save(results, f"traininglog/train_1d_segments.pt")