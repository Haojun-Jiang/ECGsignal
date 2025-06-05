import os
import torch
from torch.utils.data import Dataset
import wfdb
import numpy as np
import helper_code as hc

class ECGDataset(Dataset):
    def __init__(self, data_dir, label_map, signal_length=5000):
        self.data_dir = data_dir
        self.label_map = label_map
        self.signal_length = signal_length
        self.samples = self._collect_samples()
    
    def _collect_samples(self):
        samples = []
        for root, _, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith('.hea'):
                    base = f[:-4]
                    hea_path = os.path.join(root, base + '.hea')
                    mat_path = os.path.join(root, base + '.mat')
                    if os.path.isfile(mat_path):
                        samples.append((hea_path, mat_path))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        header_path, mat_path = self.samples[idx]

        # load signal
        signal = hc.load_recording(mat_path)  # shape (12, 5000)

        # truncate or pad
        if signal.shape[1] > self.signal_length:
            signal = signal[:, :self.signal_length]
        elif signal.shape[1] < self.signal_length:
            pad_len = self.signal_length - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_len)))

        # load header and parse Dx
        head = hc.load_header(header_path)
        # print(head)
        label = hc.get_labels(head, self.label_map)
        # print(label)

        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)