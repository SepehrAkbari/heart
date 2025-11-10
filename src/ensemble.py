from setup import *
from preprocess import y_test
from nn_dataset import DEVICE, NUM_CLASSES, INPUT_SIZE, test_loader
from nn_models import EnhancedResnetTransfer, AugmentedCNN, DFFN, EnhancedCNN

import torch
from torch.nn.functional import softmax
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

model_dir = '../model'
FIG_EVAL_DIR = f'{FIG_DIR}/model_choice'
os.makedirs(FIG_EVAL_DIR, exist_ok=True)

BEST_MODELS = {
    'EnhancedResnetTransfer': f'{model_dir}/EnhancedResnetTransfer/EnhancedResnetTransfer.pth',
    'AugmentedCNN': f'{model_dir}/AugmentedCNN/AugmentedCNN.pth',
    'DFFN': f'{model_dir}/DFFN/DFFN.pth',
    'EnhancedCNN': f'{model_dir}/EnhancedCNN/EnhancedCNN.pth'
}

def get_preds(model, data_loader):
    """
    Returns raw probability outputs for a given model.
    """
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = softmax(outputs, dim=1)
            all_outputs.append(probs.cpu().numpy())
    return np.concatenate(all_outputs, axis=0)

def vote(model_paths, test_loader, input_size, num_classes):
    """
    Performs ensemble voting.
    """
    preds = {}
    for name, path in model_paths.items():
        if name == 'EnhancedResnetTransfer':
            model = EnhancedResnetTransfer(input_size, num_classes).to(DEVICE)
        elif name == 'AugmentedCNN':
            model = AugmentedCNN(input_size, num_classes).to(DEVICE)
        elif name == 'DFFN':
            model = DFFN(input_size, num_classes).to(DEVICE)
        elif name == 'EnhancedCNN':
            model = EnhancedCNN(input_size, num_classes).to(DEVICE)
        else:
            continue
            
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        preds[name] = get_preds(model, test_loader)
    
    prob_list = list(preds.values())
    avg_probs = np.mean(prob_list, axis=0)
        
    return np.argmax(avg_probs, axis=1)

ensemble_preds = vote(BEST_MODELS, test_loader, INPUT_SIZE, NUM_CLASSES)

f2_ensemble = fbeta_score(y_test, ensemble_preds, beta=2)
print(f"Ensemble F-Beta Score: {f2_ensemble*100:.2f}%")
print(classification_report(
    y_test, 
    ensemble_preds, 
    target_names=['Normal (0)', 'Abnormal (1)'],
    labels=[0, 1],
    zero_division=0
))

prob_list = []
for name, path in BEST_MODELS.items():
    if name == 'EnhancedResnetTransfer':
        model = EnhancedResnetTransfer(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
    elif name == 'AugmentedCNN':
        model = AugmentedCNN(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
    elif name == 'DFFN':
        model = DFFN(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
    elif name == 'EnhancedCNN':
        model = EnhancedCNN(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
    else:
        continue

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    prob = get_preds(model, test_loader)
    prob_list.append(prob)

ensemble_probs = np.mean(prob_list, axis=0)

cm = confusion_matrix(y_test, ensemble_preds)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

vis.plot(cmap="Grays", values_format='d', colorbar=False, ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].yaxis.set_tick_params(rotation=90)
axes[0].tick_params(axis='x')

fpr, tpr, thresholds = roc_curve(y_test, ensemble_probs[:, 1])
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='tab:red', lw=2, label=f'Ensemble ROC curve (area = {roc_auc * 100:.1f}%)')
axes[1].plot([0, 1], [0, 1], color='tab:blue', lw=2, linestyle='--', label='Random Guessing')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.04])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Receiver Operating Characteristic')
axes[1].legend(loc='lower right')

plt.suptitle('Ensemble Model Performance', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_EVAL_DIR, 'Ensemble_performance.png'), dpi=300, bbox_inches='tight')