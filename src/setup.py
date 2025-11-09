import os
from pathlib import Path
import librosa

ROOT_DIR = str(Path.cwd().parents[0])

DATA_DIR = os.path.join(ROOT_DIR, 'data')

DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

FIG_DIR = os.path.join(ROOT_DIR, 'fig')
os.makedirs(FIG_DIR, exist_ok=True)

FIG_PREPRO_DIR = os.path.join(FIG_DIR, 'preprocessing')
os.makedirs(FIG_PREPRO_DIR, exist_ok=True)

FIG_EDA_DIR = os.path.join(FIG_DIR, 'eda')
os.makedirs(FIG_EDA_DIR, exist_ok=True)

FIG_CCV_DIR = os.path.join(FIG_DIR, 'ccv')
os.makedirs(FIG_CCV_DIR, exist_ok=True)

FIG_DL_DIR = os.path.join(FIG_DIR, 'deep_learning')
os.makedirs(FIG_DL_DIR, exist_ok=True)

FIG_TL_DIR = os.path.join(FIG_DIR, 'transfer_learning')
os.makedirs(FIG_TL_DIR, exist_ok=True)

TRAINING_SETS = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]

sample_paths = [
    os.path.join(DATA_DIR, 'training-a/a0001.wav'), 
    os.path.join(DATA_DIR, 'training-b/b0001.wav'),
    os.path.join(DATA_DIR, 'training-c/c0001.wav'),
    os.path.join(DATA_DIR, 'training-d/d0001.wav'),
    os.path.join(DATA_DIR, 'training-e/e00001.wav'),
    os.path.join(DATA_DIR, 'training-f/f0001.wav')
]
names = ['A', 'B', 'C', 'D', 'E', 'F']


sample_data = []

for i, path in enumerate(sample_paths):
    y, sr = librosa.load(path, sr=4000)
    sample_data.append({'y': y, 'sr': sr, 'name': names[i]})