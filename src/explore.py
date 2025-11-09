from setup import *
from preprocess import SAMPLE_RATE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from librosa.display import specshow
import warnings
warnings.filterwarnings('ignore')

all_labels_df = pd.read_csv(f'{DATA_PROCESSED_DIR}/labels.csv')

sample_indices = ['a0001.png', 'b0001.png', 'c0001.png', 'd0001.png', 'e00001.png', 'f0001.png']
sets = ['A', 'B', 'C', 'D', 'E', 'F']
sample_images = all_labels_df[all_labels_df['Name'].isin(sample_indices)].drop_duplicates(subset='Name').reset_index(drop=True)
sample_images = sample_images.set_index('Name').loc[sample_indices].reset_index()

max_plots = min(len(sample_images), 9)

plt.figure(figsize=(10, 12))
for i in range(max_plots):
    row = sample_images.iloc[i]
    img_path = row['Image']
    label = row['Label']
    
    img = plt.imread(img_path)
    
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Sample {sets[i]} ({'Abnormal' if label == 1 else 'Normal'})")
    
for j in range(max_plots + 1, 10):
    plt.subplot(3, 3, j).set_visible(False)
    
plt.suptitle("Sample Labeled Data", fontsize=16, y=0.9)
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(os.path.join(FIG_PREPRO_DIR, 'sample_labeled_data.png'), dpi=300, bbox_inches='tight')

def vectorize_spec(path):
    """Loads a spectrogram image and converts it to a NumPy array."""
    try:
        img = Image.open(path).convert('L') # L mode for grayscale
        return np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None
    
sample_path = sample_images.iloc[0]['Image']
spec_array = vectorize_spec(sample_path)

normal_paths = all_labels_df[all_labels_df['Label'] == 0]['Image'].tolist()
abnormal_paths = all_labels_df[all_labels_df['Label'] == 1]['Image'].tolist()

normal_spectrograms = [vectorize_spec(p) for p in normal_paths if vectorize_spec(p) is not None]
abnormal_spectrograms = [vectorize_spec(p) for p in abnormal_paths if vectorize_spec(p) is not None]

mean_normal_spec = np.mean(np.stack(normal_spectrograms), axis=0)
mean_abnormal_spec = np.mean(np.stack(abnormal_spectrograms), axis=0)

difference_map = mean_abnormal_spec - mean_normal_spec

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Average Spectrogram Per Class', fontsize=16)

specshow(mean_normal_spec, sr=SAMPLE_RATE, x_axis='time', y_axis='mel', ax=axes[0], cmap='Blues')
axes[0].set_title(f'Mean Normal ({len(normal_spectrograms)} samples)')
axes[0].set_xlabel('Time Frames')
axes[0].set_ylabel('Mel Frequency Bins')

specshow(mean_abnormal_spec, sr=SAMPLE_RATE, x_axis='time', y_axis='mel', ax=axes[1], cmap='Reds')
axes[1].set_title(f'Mean Abnormal ({len(abnormal_spectrograms)} samples)')
axes[1].set_xlabel('Time Frames')

max_abs = np.max(np.abs(difference_map))
im = specshow(difference_map, sr=SAMPLE_RATE, x_axis='time', y_axis='mel', ax=axes[2], 
                              cmap='bwr', vmin=-max_abs, vmax=max_abs)
axes[2].set_title('Difference (Abnormal - Normal)')
axes[2].set_xlabel('Time Frames')
fig.colorbar(im, ax=axes[2], format='%+2.f',)

plt.grid(False)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(FIG_EDA_DIR, 'mean_spectrograms.png'), dpi=300, bbox_inches='tight')

normal_energy_profile = np.mean(mean_normal_spec, axis=1)
abnormal_energy_profile = np.mean(mean_abnormal_spec, axis=1)

plt.figure(figsize=(10, 5))

plt.plot(normal_energy_profile, label='Mean Normal', color='tab:blue', linewidth=2)
plt.plot(abnormal_energy_profile, label='Mean Abnormal', color='tab:red', linewidth=2)
plt.title('Average Frequency Energy Per Class')
plt.xlabel('Mel Frequency Bins')
plt.ylabel('Mean Energy Intensity')
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_EDA_DIR, 'mean_frequency_energy.png'), dpi=300, bbox_inches='tight')

original_durations = []
for col, row in all_labels_df.iterrows():
    wav_path = os.path.join(DATA_DIR, row['Set'], row['Name'].replace('.png', '.wav'))
    
    if os.path.exists(wav_path):
        try:
            duration = librosa.get_duration(path=wav_path, sr=SAMPLE_RATE)
            original_durations.append(duration)
        except Exception:
            continue

plt.figure(figsize=(10, 5))

fixed_duration = (128 - 1) * 512 + 2048
fixed_duration_seconds = fixed_duration / SAMPLE_RATE

plt.hist(original_durations, bins=50, color='tan', alpha=0.7)
plt.axvline(fixed_duration_seconds, 
            color='saddlebrown', linewidth=2, linestyle='--',
            label=f'Fixed Length ({fixed_duration_seconds:.1f} seconds)')
plt.title('Distribution of Original Audio Recording Durations')
plt.xlabel('Duration (Seconds)')
plt.ylabel('# of Recordings')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIG_EDA_DIR, 'durations_distribution.png'), dpi=300, bbox_inches='tight')
