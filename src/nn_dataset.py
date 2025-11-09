from setup import *
from preprocess import *

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as VT
import torchaudio.transforms as AT
from PIL import Image
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

SPLIT_DIR = '../data/splits/'
os.makedirs(SPLIT_DIR, exist_ok=True)

if not os.path.exists(f'{SPLIT_DIR}train_data.csv') and \
   not os.path.exists(f'{SPLIT_DIR}val_data.csv') and \
   not os.path.exists(f'{SPLIT_DIR}test_data.csv'):
    train_df.to_csv(f'{SPLIT_DIR}train_data.csv', index=False)
    val_df.to_csv(f'{SPLIT_DIR}val_data.csv', index=False)
    test_df.to_csv(f'{SPLIT_DIR}test_data.csv', index=False)
    
IMG_SIZE = 128 # all images are 128x128 pixels already
IMG_W, IMG_H = IMG_SIZE, IMG_SIZE
INPUT_SIZE = IMG_W * IMG_H * 1 # 1 channel since we'll convert to grayscale
NUM_CLASSES = 2 # normal and abnormal

BATCH_SIZE = 32 # since we use DataLoader, I'll use batches. 32 is arbitrary

DEVICE = '' # the device to run the model on
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
    
train_df = pd.read_csv(f'{SPLIT_DIR}train_data.csv')
val_df = pd.read_csv(f'{SPLIT_DIR}val_data.csv')
test_df = pd.read_csv(f'{SPLIT_DIR}test_data.csv')

total_samples = len(y_train)
normal_count = np.sum(y_train == 0)
abnormal_count = np.sum(y_train == 1)

weight_normal = total_samples / (2.0 * normal_count)
weight_abnormal = total_samples / (2.0 * abnormal_count)
class_weights = torch.tensor([weight_normal, weight_abnormal], dtype=torch.float32).to(DEVICE)

class SpectrogramDataset(Dataset):
    """Dataset for loading spectrogram images."""
    def __init__(self, file_paths, targets, transform=None):
        self.file_paths = file_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        image = Image.open(path).convert('L') 
        
        if self.transform:
            tensor = self.transform(image)
        else:
            tensor = torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0) / 255.0

        label = torch.tensor(self.targets[idx], dtype=torch.long)
        return tensor, label
    
transform = VT.Compose([
    VT.ToTensor(),
    VT.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = SpectrogramDataset(X_train.values, y_train.values, transform=transform)
test_dataset = SpectrogramDataset(X_test.values, y_test.values, transform=transform)
val_dataset = SpectrogramDataset(X_val, y_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


class AudioAugmentations(object):
    """Applies random frequency and time masking/shifting."""
    def __init__(self):
        self.time_masking = AT.TimeMasking(time_mask_param=8, p=0.3)
        self.freq_masking = AT.FrequencyMasking(freq_mask_param=8)
        
    def __call__(self, tensor):
        tensor = self.time_masking(tensor)
        tensor = self.freq_masking(tensor)
        if torch.rand(1) < 0.5:
             shift = torch.randint(-10, 10, (1,)).item()
             tensor = torch.roll(tensor, shifts=shift, dims=2)
        return tensor
    
train_transform = VT.Compose([
    VT.ToTensor(), 
    VT.Resize((IMG_H, IMG_W), antialias=True),
    VT.RandomRotation(5),
    VT.Normalize(mean=[0.5], std=[0.5]),
    AudioAugmentations()
])

val_test_transform = VT.Compose([
    VT.ToTensor(), 
    VT.Resize((IMG_H, IMG_W), antialias=True),
    VT.Normalize(mean=[0.5], std=[0.5])
])

train_dataset_aug = SpectrogramDataset(X_train.values, y_train.values, transform=train_transform)
val_dataset_aug = SpectrogramDataset(X_val, y_val, transform=val_test_transform)
test_dataset_aug = SpectrogramDataset(X_test, y_test, transform=val_test_transform)

train_loader_aug = DataLoader(train_dataset_aug, batch_size=BATCH_SIZE, shuffle=True)
val_loader_aug = DataLoader(val_dataset_aug, batch_size=BATCH_SIZE, shuffle=False)
test_loader_aug = DataLoader(test_dataset_aug, batch_size=BATCH_SIZE, shuffle=False)