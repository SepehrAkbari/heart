from setup import *
from nn_models import *
from nn_dataset import *
from nn_traintest_loop import evaluate_model

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


# Evaluating BaselineFNN
model_name = "BaselineFNN"
model = BaselineFNN(INPUT_SIZE, NUM_CLASSES)

state_dict = torch.load(f'../model/{model_name}/{model_name}.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
model_BaselineFNN = model

with np.load(f"../model/{model_name}/{model_name}_losses.npz") as data:
    train_loss_BaselineFNN = data["train_loss"]
    val_loss_BaselineFNN = data["val_loss"]
    
f1_BaselineFNN, rec_BaselineFNN, prec_BaselineFNN, f2_BaselineFNN, test_targets_BaselineFNN, test_preds_BaselineFNN = evaluate_model(model_BaselineFNN, test_loader, name="Test")

print(f"F-Beta Score (test set): {f2_BaselineFNN*100:.2f}%")

cm = confusion_matrix(test_targets_BaselineFNN, test_preds_BaselineFNN)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
vis.plot(cmap="Grays", values_format='d', colorbar=False, ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].yaxis.set_tick_params(rotation=90)
axes[0].tick_params(axis='x')

axes[1].plot(train_loss_BaselineFNN, label='Training', color='tab:blue')
axes[1].plot(val_loss_BaselineFNN, label='Validation', color='tab:red')
axes[1].set_title('Loss Improvement')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend(title="Loss", loc='upper right')

plt.suptitle('BaselineFNN Model Performance', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DL_DIR, 'BaselineFNN_performance.png'), dpi=300, bbox_inches='tight')


# Evaluating EnhancedFNN
model_name = "EnhancedFNN"
model = EnhancedFNN(INPUT_SIZE, NUM_CLASSES)

state_dict = torch.load(f'../model/{model_name}/{model_name}.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
model_EnhancedFNN = model

with np.load(f"../model/{model_name}/{model_name}_losses.npz") as data:
    train_loss_EnhancedFNN = data["train_loss"]
    val_loss_EnhancedFNN = data["val_loss"]

f1_EnhancedFNN, rec_EnhancedFNN, prec_EnhancedFNN, f2_EnhancedFNN, test_targets_EnhancedFNN, test_preds_EnhancedFNN = evaluate_model(model_EnhancedFNN, test_loader, name="Test")

print(f"F-Beta Score (test set): {f2_EnhancedFNN*100:.2f}%")

cm = confusion_matrix(test_targets_EnhancedFNN, test_preds_EnhancedFNN)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

vis.plot(cmap="Grays", values_format='d', colorbar=False, ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].yaxis.set_tick_params(rotation=90)
axes[0].tick_params(axis='x')

axes[1].plot(train_loss_EnhancedFNN, label='Training', color='tab:blue')
axes[1].plot(val_loss_EnhancedFNN, label='Validation', color='tab:red')
axes[1].set_title('Loss Improvement')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend(title="Loss", loc='upper right')

plt.suptitle('EnhancedFNN Model Performance', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DL_DIR, 'EnhancedFNN_performance.png'), dpi=300, bbox_inches='tight')


# Evaluating BaselineCNN
model_name = "BaselineCNN"
model = BaselineCNN(INPUT_SIZE, NUM_CLASSES)

state_dict = torch.load(f'../model/{model_name}/{model_name}.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
model_BaselineCNN = model

with np.load(f"../model/{model_name}/{model_name}_losses.npz") as data:
    train_loss_BaselineCNN = data["train_loss"]
    val_loss_BaselineCNN = data["val_loss"]
    
f1_BaselineCNN, rec_BaselineCNN, prec_BaselineCNN, f2_BaselineCNN, test_targets_BaselineCNN, test_preds_BaselineCNN = evaluate_model(model_BaselineCNN, test_loader, name="Test")

print(f"F-Beta Score (test set): {f2_BaselineCNN*100:.2f}%")

cm = confusion_matrix(test_targets_BaselineCNN, test_preds_BaselineCNN)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

vis.plot(cmap="Grays", values_format='d', colorbar=False, ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].yaxis.set_tick_params(rotation=90)
axes[0].tick_params(axis='x')

axes[1].plot(train_loss_BaselineCNN, label='Training', color='tab:blue')
axes[1].plot(val_loss_BaselineCNN, label='Validation', color='tab:red')
axes[1].set_title('Loss Improvement')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend(title="Loss", loc='upper right')

plt.suptitle('BaselineCNN Model Performance', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DL_DIR, 'BaselineCNN_performance.png'), dpi=300, bbox_inches='tight')


# Evaluating MediumCNN
model_name = "MediumCNN"
model = MediumCNN(INPUT_SIZE, NUM_CLASSES)

state_dict = torch.load(f'../model/{model_name}/{model_name}.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
model_MediumCNN = model

with np.load(f"../model/{model_name}/{model_name}_losses.npz") as data:
    train_loss_MediumCNN = data["train_loss"]
    val_loss_MediumCNN = data["val_loss"]

f1_MediumCNN, rec_MediumCNN, prec_MediumCNN, f2_MediumCNN, test_targets_MediumCNN, test_preds_MediumCNN = evaluate_model(model_MediumCNN, test_loader, name="Test")

print(f"F-Beta Score (test set): {f2_MediumCNN*100:.2f}%")

cm = confusion_matrix(test_targets_MediumCNN, test_preds_MediumCNN)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

vis.plot(cmap="Grays", values_format='d', colorbar=False, ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].yaxis.set_tick_params(rotation=90)
axes[0].tick_params(axis='x')

axes[1].plot(train_loss_MediumCNN, label='Training', color='tab:blue')
axes[1].plot(val_loss_MediumCNN, label='Validation', color='tab:red')
axes[1].set_title('Loss Improvement')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend(title="Loss", loc='upper right')

plt.suptitle('MediumCNN Model Performance', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DL_DIR, 'MediumCNN_performance.png'), dpi=300, bbox_inches='tight')


# Evaluating EnhancedCNN
model_name = "EnhancedCNN"
model = EnhancedCNN(INPUT_SIZE, NUM_CLASSES)

state_dict = torch.load(f'../model/{model_name}/{model_name}.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
model_EnhancedCNN = model

with np.load(f"../model/{model_name}/{model_name}_losses.npz") as data:
    train_loss_EnhancedCNN = data["train_loss"]
    val_loss_EnhancedCNN = data["val_loss"]
    
f1_EnhancedCNN, rec_EnhancedCNN, prec_EnhancedCNN, f2_EnhancedCNN, test_targets_EnhancedCNN, test_preds_EnhancedCNN = evaluate_model(model_EnhancedCNN, test_loader, name="Test")

print(f"F-Beta Score (test set): {f2_EnhancedCNN*100:.2f}%")

cm = confusion_matrix(test_targets_EnhancedCNN, test_preds_EnhancedCNN)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

vis.plot(cmap="Grays", values_format='d', colorbar=False, ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].yaxis.set_tick_params(rotation=90)
axes[0].tick_params(axis='x')

axes[1].plot(train_loss_EnhancedCNN, label='Training', color='tab:blue')
axes[1].plot(val_loss_EnhancedCNN, label='Validation', color='tab:red')
axes[1].set_title('Loss Improvement')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend(title="Loss", loc='upper right')

plt.suptitle('EnhancedCNN Model Performance', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DL_DIR, 'EnhancedCNN_performance.png'), dpi=300, bbox_inches='tight')


# Evaluating seCNN
model_name = "seCNN"
model = seCNN(INPUT_SIZE, NUM_CLASSES)

state_dict = torch.load(f'../model/{model_name}/{model_name}.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
model_seCNN = model

with np.load(f"../model/{model_name}/{model_name}_losses.npz") as data:
    train_loss_seCNN = data["train_loss"]
    val_loss_seCNN = data["val_loss"]
    
f1_seCNN, rec_seCNN, prec_seCNN, f2_seCNN, test_targets_seCNN, test_preds_seCNN = evaluate_model(model_seCNN, test_loader, name="Test")

print(f"F-Beta Score (test set): {f2_seCNN*100:.2f}%")

cm = confusion_matrix(test_targets_seCNN, test_preds_seCNN)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

vis.plot(cmap="Grays", values_format='d', colorbar=False, ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].yaxis.set_tick_params(rotation=90)
axes[0].tick_params(axis='x')

axes[1].plot(train_loss_seCNN, label='Training', color='tab:blue')
axes[1].plot(val_loss_seCNN, label='Validation', color='tab:red')
axes[1].set_title('Loss Improvement')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend(title="Loss", loc='upper right')

plt.suptitle('seCNN Model Performance', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DL_DIR, 'seCNN_performance.png'), dpi=300, bbox_inches='tight')


# Evaluating DFFN
model_name = "DFFN"
model = DFFN(INPUT_SIZE, NUM_CLASSES)

state_dict = torch.load(f'../model/{model_name}/{model_name}.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
model_DFFN = model

with np.load(f"../model/{model_name}/{model_name}_losses.npz") as data:
    train_loss_DFFN = data["train_loss"]
    val_loss_DFFN = data["val_loss"]
    
f1_DFFN, rec_DFFN, prec_DFFN, f2_DFFN, test_targets_DFFN, test_preds_DFFN = evaluate_model(model_DFFN, test_loader, name="Test")

print(f"F-Beta Score (test set): {f2_DFFN*100:.2f}%")

cm = confusion_matrix(test_targets_DFFN, test_preds_DFFN)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

vis.plot(cmap="Grays", values_format='d', colorbar=False, ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].yaxis.set_tick_params(rotation=90)
axes[0].tick_params(axis='x')

axes[1].plot(train_loss_DFFN, label='Training', color='tab:blue')
axes[1].plot(val_loss_DFFN, label='Validation', color='tab:red')
axes[1].set_title('Loss Improvement')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend(title="Loss", loc='upper right')

plt.suptitle('DFFN Model Performance', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DL_DIR, 'DFFN_performance.png'), dpi=300, bbox_inches='tight')


# Evaluating AugmentedCNN
model_name = "AugmentedCNN"
model = AugmentedCNN(INPUT_SIZE, NUM_CLASSES)

state_dict = torch.load(f'../model/{model_name}/{model_name}.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
model_AugmentedCNN = model

with np.load(f"../model/{model_name}/{model_name}_losses.npz") as data:
    train_loss_AugmentedCNN = data["train_loss"]
    val_loss_AugmentedCNN = data["val_loss"]
    
f1_AugmentedCNN, rec_AugmentedCNN, prec_AugmentedCNN, f2_AugmentedCNN, test_targets_AugmentedCNN, test_preds_AugmentedCNN = evaluate_model(model_AugmentedCNN, test_loader, name="Test")

print(f"F-Beta Score (test set): {f2_AugmentedCNN*100:.2f}%")

cm = confusion_matrix(test_targets_AugmentedCNN, test_preds_AugmentedCNN)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

vis.plot(cmap="Grays", values_format='d', colorbar=False, ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].yaxis.set_tick_params(rotation=90)
axes[0].tick_params(axis='x')

axes[1].plot(train_loss_AugmentedCNN, label='Training', color='tab:blue')
axes[1].plot(val_loss_AugmentedCNN, label='Validation', color='tab:red')
axes[1].set_title('Loss Improvement')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend(title="Loss", loc='upper right')

plt.suptitle('AugmentedCNN Model Performance', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DL_DIR, 'AugmentedCNN_performance.png'), dpi=300, bbox_inches='tight')


# Evaluating BaselineResnetTransfer
model_name = "BaselineResnetTransfer"
model = BaselineResnetTransfer(INPUT_SIZE, NUM_CLASSES)

state_dict = torch.load(f'../model/{model_name}/{model_name}.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
model_BaselineResnetTransfer = model

with np.load(f"../model/{model_name}/{model_name}_losses.npz") as data:
    train_loss_BaselineResnetTransfer = data["train_loss"]
    val_loss_BaselineResnetTransfer = data["val_loss"]
    
f1_BaselineResnetTransfer, rec_BaselineResnetTransfer, prec_BaselineResnetTransfer, f2_BaselineResnetTransfer, test_targets_BaselineResnetTransfer, test_preds_BaselineResnetTransfer = evaluate_model(model_BaselineResnetTransfer, test_loader, name="Test")

print(f"F-Beta Score (test set): {f2_BaselineResnetTransfer*100:.2f}%")

cm = confusion_matrix(test_targets_BaselineResnetTransfer, test_preds_BaselineResnetTransfer)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

vis.plot(cmap="Grays", values_format='d', colorbar=False, ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].yaxis.set_tick_params(rotation=90)
axes[0].tick_params(axis='x')

axes[1].plot(train_loss_BaselineResnetTransfer, label='Training', color='tab:blue')
axes[1].plot(val_loss_BaselineResnetTransfer, label='Validation', color='tab:red')
axes[1].set_title('Loss Improvement')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend(title="Loss", loc='upper right')

plt.suptitle('BaselineResnetTransfer Model Performance', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_TL_DIR, 'BaselineResnetTransfer_performance.png'), dpi=300, bbox_inches='tight')


# Evaluating EnhancedResnetTransfer
model_name = "EnhancedResnetTransfer"
model = EnhancedResnetTransfer(INPUT_SIZE, NUM_CLASSES)

state_dict = torch.load(f'../model/{model_name}/{model_name}.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
model_EnhancedResnetTransfer = model

with np.load(f"../model/{model_name}/{model_name}_losses.npz") as data:
    train_loss_EnhancedResnetTransfer = data["train_loss"]
    val_loss_EnhancedResnetTransfer = data["val_loss"]
    
f1_EnhancedResnetTransfer, rec_EnhancedResnetTransfer, prec_EnhancedResnetTransfer, f2_EnhancedResnetTransfer, test_targets_EnhancedResnetTransfer, test_preds_EnhancedResnetTransfer = evaluate_model(model_EnhancedResnetTransfer, test_loader, name="Test")

print(f"F-Beta Score (test set): {f2_EnhancedResnetTransfer*100:.2f}%")

cm = confusion_matrix(test_targets_EnhancedResnetTransfer, test_preds_EnhancedResnetTransfer)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

vis.plot(cmap="Grays", values_format='d', colorbar=False, ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].yaxis.set_tick_params(rotation=90)
axes[0].tick_params(axis='x')

axes[1].plot(train_loss_EnhancedResnetTransfer, label='Training', color='tab:blue')
axes[1].plot(val_loss_EnhancedResnetTransfer, label='Validation', color='tab:red')
axes[1].set_title('Loss Improvement')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend(title="Loss", loc='upper right')

plt.suptitle('EnhancedResnetTransfer Model Performance', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_TL_DIR, 'EnhancedResnetTransfer_performance.png'), dpi=300, bbox_inches='tight')