import os
import numpy as np
import pandas as pd
import pywt as pw
import matplotlib.pyplot as plt
from scipy.signal import stft
from tqdm import tqdm
from sklearn.utils import resample

def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# (Re)sort leads using the standard order of leads for the standard twelve-lead ECG.
def sort_leads(leads):
    x = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    leads = sorted(leads, key=lambda lead: (x.index(lead) if lead in x else len(x) + leads.index(lead)))
    return tuple(leads)

# Find header and recording files.
def find_challenge_files(data_directory):
    header_files = list()
    recording_files = list()
    for f in os.listdir(data_directory):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension=='.hea':
            header_file = os.path.join(data_directory, root + '.hea')
            recording_file = os.path.join(data_directory, root + '.mat')
            if os.path.isfile(header_file) and os.path.isfile(recording_file):
                header_files.append(header_file)
                recording_files.append(recording_file)
    return header_files, recording_files

# Load header file as a string.
def load_header(header_file):
    with open(header_file, 'r') as f:
        header = f.read()
    return header

# Load recording file as an array.
def load_recording(recording_file, header=None, leads=None,key='val'):
    from scipy.io import loadmat
    recording = loadmat(recording_file)[key]
    if header and leads:
        recording = choose_leads(recording, header, leads)
    return recording

# Choose leads from the recording file.
def choose_leads(recording, header, leads):
    num_leads = len(leads)
    num_samples = np.shape(recording)[1]
    chosen_recording = np.zeros((num_leads, num_samples), recording.dtype)
    available_leads = get_leads(header)
    for i, lead in enumerate(leads):
        if lead in available_leads:
            j = available_leads.index(lead)
            chosen_recording[i, :] = recording[j, :]
    return chosen_recording

# Get recording ID.
def get_recording_id(header):
    recording_id = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                recording_id = l.split(' ')[0]
            except:
                pass
        else:
            break
    return recording_id

# Get leads from header.
def get_leads(header):
    leads = list()
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            leads.append(entries[-1])
        else:
            break
    return tuple(leads)

# Get age from header.
def get_age(header):
    age = None
    for l in header.split('\n'):
        if l.startswith('#Age'):
            try:
                age = float(l.split(': ')[1].strip())
            except:
                age = float('nan')
    return age

# Get sex from header.
def get_sex(header):
    sex = None
    for l in header.split('\n'):
        if l.startswith('#Sex'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex

# Get number of leads from header.
def get_num_leads(header):
    num_leads = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                num_leads = float(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_leads

# Get frequency from header.
def get_frequency(header):
    frequency = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                frequency = float(l.split(' ')[2])
            except:
                pass
        else:
            break
    return frequency

# Get number of samples from header.
def get_num_samples(header):
    num_samples = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                num_samples = float(l.split(' ')[3])
            except:
                pass
        else:
            break
    return num_samples

# Get analog-to-digital converter (ADC) gains from header.
def get_adc_gains(header, leads):
    adc_gains = np.zeros(len(leads))
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    adc_gains[j] = float(entries[2].split('/')[0])
                except:
                    pass
        else:
            break
    return adc_gains

# Get baselines from header.
def get_baselines(header, leads):
    baselines = np.zeros(len(leads))
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    baselines[j] = float(entries[4].split('/')[0])
                except:
                    pass
        else:
            break
    return baselines

# Get labels from header.
def get_labels(header):
    labels = list()
    scored_labels = np.asarray(pd.read_csv('D:\study\Msc project\project\scr\dx_mapping_scored.csv').iloc[:,1], dtype="str")
    for l in header.split('\n'):
        if l.startswith('# Dx'):
            try:
                entries = l.split(': ')[1].split(',')
                for entry in entries:
                    if any(j == entry for j in scored_labels):
                        labels.append(entry.strip())
            except:
                pass
    return map_class_to_single(labels)

def map_class_to_single(labels):
    AF_DX = {'164889003'}
    NOMAL_DX = {'426783006'}
    OTHER_DX = {'39732003','251146004','10370003','365413008','164917005','47665007','164934002','59931005'}

    if(len(labels)==0):
        return 2       # other
    elif any(j in AF_DX for j in labels):
        return 1        # AF
    elif any(j in NOMAL_DX for j in labels):
        return 0        # Normal
    elif any(j in OTHER_DX for j in labels):
        return 2        # other
    else:
        return 3        # other diseases
# get QRS group
def qrs_detection(signal, sample_rate=500, max_bpm=300):
    coeffs = pw.swt(signal, wavelet="haar", level=2, start_level=0, axis=-1)
    d2 = coeffs[1][1]  # 2nd level detail coefficients

    avg = np.mean(d2)
    std = np.std(d2)
    sig_thres = [abs(i) if abs(i) > 2.0 * std else 0 for i in d2 - avg]

    window = int((60.0 / max_bpm) * sample_rate)
    sig_len = len(signal)
    # print(len(signal))
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

# get segmants
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
        if len(segment) == length:
            segments.append(segment)

    segments = np.array(segments)
    labels = np.repeat(label, segments.shape[0])
    return np.hstack((segments, labels[:, np.newaxis]))

def collect_samples(data_dirs):
    samples = []
    for data_dir in data_dirs:
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

def segments(data_path):
    segments = []
    samples_path = collect_samples(data_path)
    for head_path, mat_path in samples_path:
        head = load_header(head_path)
        label = get_labels(head)
        raw_signal = load_recording(mat_path, head, leads = ['II'])
        raw_signal = np.array(raw_signal[0])
        rpeaks = qrs_detection(raw_signal, sample_rate=500, max_bpm=300)
        seg = get_segments(raw_signal, rpeaks, label, length=1000)
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
    return segments

def get_spectrogram(segments, output_dir = './data/spectrogram', fs = 500):
    os.makedirs(output_dir, exist_ok=True)
    print(-1)
    for idx, row in tqdm(enumerate(segments), total=len(segments)):
        signal = row[:1000]
        label = int(row[1000])
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        f, t, Zxx = stft(signal, fs=fs, nperseg=128, noverlap=64)
        spectrogram = np.abs(Zxx)
        print(0)
        plt.figure(figsize=(2.5, 2.5))
        plt.pcolormesh(t, f, spectrogram, shading='gouraud')
        plt.axis('off') 
        plt.tight_layout(pad=0)

        save_path = os.path.join(label_dir, f'{idx}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(1)

    return output_dir

def resample_data(samples):
    samples_0 = [s for s in samples if s[1000] == 0]
    samples_1 = [s for s in samples if s[1000] == 1]
    samples_2 = [s for s in samples if s[1000] == 2]
    samples_3 = [s for s in samples if s[1000] == 3]
    samples_2_down = resample(samples_2, replace=False, n_samples=40000, random_state=42)
    samples_3_down = resample(samples_3, replace=False, n_samples=40000, random_state=42)
    samples_0_down = resample(samples_0, replace=False, n_samples=40000, random_state=42)
    samples_1_up = resample(samples_1, replace=True, n_samples=40000, random_state=42)
    balanced_samples = samples_0_down + samples_1_up + samples_2_down + samples_3_down
    balanced_samples = np.array(balanced_samples)
    return balanced_samples