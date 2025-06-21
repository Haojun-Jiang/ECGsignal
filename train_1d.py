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
        segments = []
        samples_path = hc.collect_samples(['data/training/chapman_shaoxing','data/training/ningbo'])
        for head_path, mat_path in samples_path:
            head = hc.load_header(head_path)
            label = hc.get_labels(head)
            raw_signal = hc.load_recording(mat_path, head, leads = ['II'])
            raw_signal = np.array(raw_signal[0])
            rpeaks = hc.qrs_detection(raw_signal, sample_rate=500, max_bpm=300)
            seg = hc.get_segments(raw_signal, rpeaks, label, length=1000)
            if(seg is None):
                continue
            else:
                print(seg.shape)
            for s in seg:
                if(s is None):
                    continue
                else:
                    segments.append([s[:999], s[1000]])
        segments = np.array(segments)
        np.save('data/segments.npy', segments)

    raw_data = ECGsegments('data/segments.npy', segments=segments, segment_length=1000)

    train_dataset, val_dataset = random_split(raw_data, [int(len(raw_data)*0.8), len(raw_data)-int(len(raw_data)*0.8)])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

    results = train_module(cnn_feed_rnn, train_loader, val_loader, optimizer, criterion, device='cuda',n_epochs=2)
    return results

results = train_1d_segments()
os.makedirs('./traininglog', exist_ok=True)
torch.save(results, f"traininglog/train_1d_segments.pt")