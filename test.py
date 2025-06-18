from data import ECGDataset
from torch.utils.data import DataLoader
from collections import Counter
import pandas as pd

# print(pd.read_csv('D:\study\Msc project\project\scr\dx_mapping_scored.csv'))
def load_label_map(txt_path):
    with open(txt_path) as f:
        codes = [line.strip() for line in f if line.strip()]
    return {code: i for i, code in enumerate(codes)}

label_map = load_label_map('D:\study\Msc project\project\scr\classes.txt')

# 构建 Dataset 实例
dataset = ECGDataset(
    data_dirs=['data/training/chapman_shaoxing','data/training/ningbo'],
    label_map=label_map,
    signal_length=5000,
    leads=["II"]
)
# 检查样本数量


# 获取单个样本
# for signal, label in dataset:
#     print(signal)
#     break

# 构建 DataLoader
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 遍历一个 batch
for batch_signals, batch_labels in loader:
    print("Batch signals:", batch_signals.shape)  # (8, 12, 5000)
    print("Batch labels:", batch_labels)    # (8, num_classes)