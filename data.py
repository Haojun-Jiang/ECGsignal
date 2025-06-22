import os
import torch
from torch.utils.data import Dataset
import wfdb
import numpy as np
import helper_code as hc
from PIL import Image

class ECGDataset(Dataset):
    def __init__(self, data_dirs, label_map, leads, signal_length=5000):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.data_dirs = data_dirs
        self.label_map = label_map
        self.signal_length = signal_length
        self.samples = self._collect_samples()
        self.leads = leads
    
    def _collect_samples(self):
        samples = []
        for data_dir in self.data_dirs:
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f.endswith('.hea'):
                        base = f[:-4]
                        hea_path = os.path.join(root, base + '.hea')
                        mat_path = os.path.join(root, base + '.mat')
                        if os.path.exists(mat_path):
                            samples.append((hea_path, mat_path))
                        # samples.append((hea_path, mat_path))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        header_path, mat_path = self.samples[idx]
        # load header and parse Dx
        head = hc.load_header(header_path)
        # print(head)
        label = hc.get_labels(head)
        # print(label)

        # load signal
        signal = self.Normalize(hc.load_recording(mat_path, header=head, leads = self.leads))  # shape (1, 5000)
        signal = self.scaling(signal)
        # truncate or pad
        if signal.shape[1] > self.signal_length:
            signal = signal[:, :self.signal_length]
        elif signal.shape[1] < self.signal_length:
            pad_len = self.signal_length - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_len)))

        
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

    def Normalize(self, signal):
        signal = np.asarray(signal)
        if signal.size == 0:
            raise ValueError("Input is empty")
        
        std = np.std(signal)
        if std == 0:
            return signal - np.mean(signal)
        
        return (signal - np.mean(signal)) / std

    def scaling(self, signal):
        return signal * 1.2


class ECGsegments(Dataset):
    def __init__(self, data_path, segments,segment_length=1000):
        if isinstance(data_path, str):
            data_dir = [data_path]
        self.data_path = data_dir
        self.segment_length = segment_length
        self.samples = segments
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        signal, label = self.samples[idx][:1000], self.samples[idx][1000]
        signal = self.Normalize(signal)
        signal = self.scaling(signal)
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.int64)
        return signal_tensor, label_tensor
    
    def Normalize(self, signal):
        signal = np.asarray(signal)
        if signal.size == 0:
            raise ValueError("Input is empty")
        
        std = np.std(signal)
        if std == 0:
            return signal - np.mean(signal)
        
        return (signal - np.mean(signal)) / std
    
    def scaling(self, signal):
        return signal * 1.2

# spectrogram
class STFTgraph (Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform

        self.images = []
        self.labels = []
        self.class_to_index = {}

        classes = sorted(os.listdir(root_dir))
        for i, class_name in enumerate(classes):
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                self.class_to_index[class_name] = i
                for image_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, image_name)
                    self.images.append(img_path)
                    self.labels.append(class_name)

        self.classes = sorted(set(self.labels))
        self.class_to_index = {class_name: i for i, class_name in enumerate(self.classes)}

    def __len__ (self):
        return len(self.images)
    
    def __getitem__ (self, index):
        img_path = self.images[index]
        label = self.class_to_index[self.labels[index]]
        image = Image.open(img_path).convert('L').resize((256, 256))

        if self.transform:
            image = self.transform(image)

        return image, label