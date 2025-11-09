from setup import *

import os
import matplotlib.pyplot as plt
from librosa.display import waveshow, specshow
from librosa.feature import melspectrogram
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

fig, axs = plt.subplots(2, 3, figsize=(18, 8))
audio_data = [(x['y'], x['sr'], x['name']) for x in sample_data]
for i, (y, sr, label) in enumerate(audio_data):
    row = i // 3
    col = i % 3
    waveshow(y, sr=sr, ax=axs[row, col], axis='time', color='tab:red')
    axs[row, col].set_title(f"Sample {label}")
    axs[row, col].set_xlabel("Time (s)")
    axs[row, col].set_ylabel("Amplitude")
    axs[row, col].grid(True, linestyle='--', alpha=0.5)
    
plt.suptitle("Waveforms of Sample Audio Clips", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_PREPRO_DIR, 'sample_waveforms.png'), dpi=300, bbox_inches='tight')

SAMPLE_RATE = 4000 # samples per second (Hz)
DURATION = 20 # duration of each audio clip (seconds)
N_MELS = 128 # number of Mel bands (height of spectrogram)
N_FFT = 2048 # FFT window size
HOP_LENGTH = 512 # number of samples between frames
FIXED_LENGTH = 128 # fixed time frames (width of spectrogram)

def fix_length(y, fixed_length, hop_length, n_fft):
    """
    Adjust the length of the audio signal `y` to ensure it results in a spectrogram with `fixed_length` time frames.
    """
    goal = (fixed_length - 1) * hop_length + n_fft
    
    if len(y) > goal:
        # Truncate
        y_out = y[:goal]
    elif len(y) < goal:
        # Pad (with zeros)
        padding_needed = goal - len(y)
        y_out = np.pad(y, (0, padding_needed), mode='constant')
    else:
        y_out = y
    return y_out

plt.figure(figsize=(15, 15))

for i, data in enumerate(sample_data):
    ax = plt.subplot(3, 3, i + 1)
    
    y = data['y']
    sr = data['sr']
    name = data['name']
    
    y_fixed = fix_length(y, FIXED_LENGTH, HOP_LENGTH, N_FFT)
    
    mel_spectrogram = melspectrogram(
        y=y_fixed,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    specshow(
        mel_spectrogram_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis='time',
        y_axis='mel',
        cmap='Reds',
        ax=ax
    )
    
    plt.title(f'Sample {name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency (Hz)')

for j in range(len(sample_data) + 1, 10):
    plt.subplot(3, 3, j).set_visible(False)

plt.suptitle(f'Mel-Spectrograms of Sample Audio Clips', 
             fontsize=16, y=0.92)
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(os.path.join(FIG_PREPRO_DIR, 'sample_melspectrograms.png'), dpi=300, bbox_inches='tight')

DPI = 100
FIGSIZE = (N_MELS / DPI, FIXED_LENGTH / DPI)

def make_melspec(audio_path, output_path):
    """
    Loads and generate Mel-spectrograms.
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        y_fixed = fix_length(y, FIXED_LENGTH, HOP_LENGTH, N_FFT)

        mel_spectrogram = melspectrogram(
            y=y_fixed, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        plt.figure(figsize=FIGSIZE, dpi=DPI)
        
        specshow(mel_spectrogram_db, cmap='plasma')
        
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return True

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False
    
def read_labels():
    """
    Reads all annotation files and combines them into a single DataFrame.
    """
    all_labels = []
    
    training_annotation_files = glob.glob(os.path.join(DATA_DIR, 'training-*', 'REFERENCE.csv'))
    val_annotation_file = os.path.join(DATA_DIR, 'validation', 'REFERENCE.csv')
    
    for file_path in training_annotation_files:
        try:
            df = pd.read_csv(file_path, header=None, names=['Name', 'Label'])
            df['Set'] = os.path.basename(os.path.dirname(file_path))
            all_labels.append(df)
        except Exception as e:
            print(f"Could not read {file_path}. Error: {e}")
    
    try:
        df_val = pd.read_csv(val_annotation_file, header=None, names=['Name', 'Label'])
        df_val['Set'] = 'validation'
        all_labels.append(df_val)
    except Exception as e:
        print(f"Warning: Could not read validation annotation file. Error: {e}")

    if all_labels:
        master_df = pd.concat(all_labels, ignore_index=True)
        master_df['Name'] = master_df['Name'].apply(lambda x: x + '.png')
        return master_df
    else:
        print("ERROR: No labels collected from any file.")
        return pd.DataFrame()
    
labels_df = read_labels()

def process_data():
    """
    Process all audio files and labels.
    """
    data_subfolders = glob.glob(os.path.join(DATA_DIR, 'training-*'))
    validation_folder = os.path.join(DATA_DIR, 'validation')
    
    if os.path.isdir(validation_folder):
        data_subfolders.append(validation_folder)

    if not data_subfolders:
        print("No subdirectories found.")
        return None, None

    processed = 0
    
    for folder_idx, subfolder_path in enumerate(data_subfolders):
        subfolder_name = os.path.basename(subfolder_path)
        output_subfolder = os.path.join(DATA_PROCESSED_DIR, subfolder_name)
        os.makedirs(output_subfolder, exist_ok=True)
        
        wav_files = glob.glob(os.path.join(subfolder_path, '*.wav'))
        
        print(f"\nProcessing Folder #{folder_idx + 1}: {subfolder_name} ({len(wav_files)} files)")

        for wav_count, wav_path in enumerate(wav_files):
            if (wav_count + 1) % 100 == 0:
                print(f"Processed {wav_count + 1} of {len(wav_files)}")
            
            file_name = os.path.basename(wav_path)
            output_file_name = file_name.replace('.wav', '.png')
            output_path = os.path.join(output_subfolder, output_file_name)
            
            if make_melspec(wav_path, output_path): 
                processed += 1

    global_labels_df = read_labels()
    
    labels_output_path = os.path.join(DATA_PROCESSED_DIR, 'labels_raw.csv')
    global_labels_df.to_csv(labels_output_path, index=False)
    
    print(f"\nTotal audio files processed: {processed}")
    print(f"Total labels collected: {len(global_labels_df)}")
    print(f"Labels saved to: {labels_output_path}")

    return global_labels_df, DATA_PROCESSED_DIR

labels, processed_dir = process_data()

LABELS_PATH = os.path.join(DATA_PROCESSED_DIR, 'labels_raw.csv')
labels_df = pd.read_csv(LABELS_PATH)

def get_full_path(row):
    subfolder = row['Set']
    filename = row['Name']
    return os.path.join(DATA_PROCESSED_DIR, subfolder, filename)

labels_df['Image'] = labels_df.apply(get_full_path, axis=1)
labels_df['Label'] = labels_df['Label'].map({-1: 0, 1: 1})

FINAL_DATA_PATH = os.path.join(DATA_PROCESSED_DIR, 'labels.csv')
labels_df.to_csv(FINAL_DATA_PATH, index=False)

LABELS_PATH = os.path.join(DATA_PROCESSED_DIR, 'labels.csv')
labels_df = pd.read_csv(LABELS_PATH)

plt.figure(figsize=(5, 4))
labels_df['Label'].value_counts().plot(kind='bar', color='tab:red')
plt.title('Class Distribution (0: Normal, 1: Abnormal)')
plt.xlabel('Label')
plt.ylabel('# of Samples')
plt.grid(axis='y', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(FIG_EDA_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')

validation_df = labels_df[labels_df['Set'] == 'validation'].copy()
training_and_test_df_raw = labels_df[labels_df['Set'].str.startswith('training')].copy()

validation_filenames = set(validation_df['Name'])

training_and_test_df = training_and_test_df_raw[
    ~training_and_test_df_raw['Name'].isin(validation_filenames)
].copy()

removed_count = len(training_and_test_df_raw) - len(training_and_test_df)

X = training_and_test_df['Image']
y = training_and_test_df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.22, 
    random_state=325, 
    stratify=y
)

train_df = pd.DataFrame({'Image': X_train, 'Label': y_train}).reset_index(drop=True)
test_df = pd.DataFrame({'Image': X_test, 'Label': y_test}).reset_index(drop=True)
val_df = validation_df[['Image', 'Label']].reset_index(drop=True)

X_val = val_df['Image']
y_val = val_df['Label']