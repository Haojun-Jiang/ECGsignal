from data import ECGDataset
from torch.utils.data import DataLoader

def load_label_map(txt_path):
    with open(txt_path) as f:
        codes = [line.strip() for line in f if line.strip()]
    return {code: i for i, code in enumerate(codes)}

label_map = load_label_map('D:\study\Msc project\project\scr\classes.txt')

# 构建 Dataset 实例
dataset = ECGDataset(
    data_dirs=['data/training/chapman_shaoxing','data/training/ningbo'],
    label_map=label_map,
    signal_length=5000
)
# 检查样本数量
print("Total samples:", len(dataset))

# 获取单个样本
signal, label = dataset[0]
print("Signal:", signal)   # 应该是 (12, 5000)
print("Label:", label)

# 构建 DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 遍历一个 batch
# for batch_signals, batch_labels in loader:
#     print("Batch signals:", batch_signals.shape)  # (8, 12, 5000)
#     print("Batch labels:", batch_labels.shape)    # (8, num_classes)
#     break