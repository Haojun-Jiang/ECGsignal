from data import ECGDataset
from torch.utils.data import DataLoader
from collections import Counter
import helper_code as hc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
import pywt as pw 


# print(pd.read_csv('D:\study\Msc project\project\scr\dx_mapping_scored.csv'))
def load_label_map(txt_path):
    with open(txt_path) as f:
        codes = [line.strip() for line in f if line.strip()]
    return {code: i for i, code in enumerate(codes)}

label_map = load_label_map('D:\study\Msc project\project\scr\classes.txt')

samples_path = hc.collect_samples(['data/training/chapman_shaoxing'])

head_path, mat_path = samples_path[0]

head = hc.load_header(head_path)
label = hc.get_labels(head)
signal_1 = hc.load_recording(mat_path, header=head, leads = ['II'])  # shape (1, 5000)
signal_1 = np.array(signal_1[0])
# r_peaks = hc.qrs_detect(signal)

# segments = hc.get_segments(signal,r_peaks, label, 1000)
# print(segments.shape)


# 假设这是你的一条 ECG 信号
# signal = np.random.randn(5000)  # 用随机数代替，实际用你的真实信号

# R-peaks（来自你提供的图像）
# rpeaks = np.array([0, 229, 466, 732, 966, 1244, 1512, 1803, 2077, 2336,
#                    2576, 2856, 3121, 3393, 3583, 3851, 4069, 4343, 4583])

# 标签（例如当前信号为房颤：label = 1）
# label = 1
# print(signal)
# segment函数
def get_segments(signal, rpeaks, label, length=1000):
    n = rpeaks.shape[0]
    if n <= 8:
        return None

    segments = []

    for i in range(2, n - 6):
        l, r = rpeaks[i], rpeaks[i + 3]
        padding = length - (r - l)
        if padding % 2 == 0:
            l_padding = r_padding = padding // 2
        else:
            l_padding = (padding - 1) // 2
            r_padding = (padding + 1) // 2

        if l_padding > l:
            r_padding += l_padding - l
            l_padding = l

        if r + r_padding >= signal.shape[0]:
            r_padding = signal.shape[0] - 1 - r
            l_padding = l - signal.shape[0] + 1 + length

        segment = signal[l - l_padding : r + r_padding].copy()
        print(len(segment))
        if len(segment) == length:
            segments.append(segment)

    segments = np.array(segments)
    labels = np.repeat(label, segments.shape[0])
    return np.hstack((segments, labels[:, np.newaxis]))


# # 获取结果
# segments_with_labels = get_segments(signal, r_peaks, label, length=1000)

# # 查看结果
# print("生成的样本数量：", segments_with_labels.shape[0])
# print("每条样本的 shape：", segments_with_labels.shape[1]) 

def qrs_detection(signal, sample_rate=500, max_bpm=300):
    coeffs = pw.swt(signal, wavelet="haar", level=2, start_level=0, axis=-1)
    d2 = coeffs[1][1]  # 2nd level detail coefficients

    avg = np.mean(d2)
    std = np.std(d2)
    sig_thres = [abs(i) if abs(i) > 2.0 * std else 0 for i in d2 - avg]

    window = int((60.0 / max_bpm) * sample_rate)
    sig_len = len(signal)
    print(len(signal))
    n_windows = int(sig_len / window)
    modulus, qrs = [], []

    for i in range(n_windows):
        start = i * window
        end = min([(i + 1) * window, sig_len])
        mx = max(sig_thres[start:end])
        if mx > 0:
            modulus.append((start + np.argmax(sig_thres[start:end]), mx))

    merge_width = int(0.2 * sample_rate)
    i = 0
    while i < len(modulus) - 1:
        ann = modulus[i][0]
        if modulus[i + 1][0] - modulus[i][0] < merge_width:
            if modulus[i + 1][1] > modulus[i][1]:
                ann = modulus[i + 1][0]
            i += 1
        qrs.append(ann)
        i += 1

    window_check = int(sample_rate / 6)
    r_peaks = [0] * len(qrs)

    for i, loc in enumerate(qrs):
        start = max(0, loc - window_check)
        end = min(sig_len, loc + window_check)
        wdw = np.absolute(signal[start:end] - np.mean(signal[start:end]))
        pk = np.argmax(wdw)
        r_peaks[i] = start + pk

    return np.array(r_peaks)

# =======================
# ✅ 模拟 ECG 数据测试
# =======================
fs = 500  # 采样率
duration = 10  # 秒
t = np.linspace(0, 15, fs * duration)
# 使用简单的正弦和 sawtooth 波模仿心跳波形
# signal = 0.6 * np.sin(2 * np.pi * 1.3 * t) + 0.4 * sawtooth(2 * np.pi * 1.3 * t)
# print(signal)

# 添加几个大的 QRS-like 峰值（模拟R波）
# for loc in range(100, len(signal), 300):
#     signal[loc:loc+3] += 1.5

# 调用你的函数
rpeaks = qrs_detection(signal_1, sample_rate=fs)

segments = hc.get_segments(signal_1, rpeaks, label, length=1000)

# =======================
# ✅ 可视化结果
# =======================
# plt.figure(figsize=(12, 4))
# plt.plot(t, signal_1, label='ECG signal')
# plt.plot(t[rpeaks], signal_1[rpeaks], 'ro', label='Detected R-peaks')
# plt.title('QRS Detection Verification')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()