from setup import *
from preprocess import train_df, test_df, val_df
from explore import sample_images

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             recall_score,
                             precision_score,
                             f1_score)
import warnings
warnings.filterwarnings('ignore')

def mean_img(image_path):
    """
    Extracts the overall average energy from the spectrogram.
    """
    try:
        img = Image.open(image_path).convert('L')
        spec_array = np.array(img, dtype=np.float32)
    except Exception:
        return 0.0 

    return np.mean(spec_array) # overall average energy

def intgrl_img(image_path, n_mels=128):
    """
    Extracts the average energy from the high-frequency half of the spectrogram.
    """
    try:
        img = Image.open(image_path).convert('L')
        spec_array = np.array(img, dtype=np.float32)
    except Exception:
        return 0.0 

    top_half = spec_array[n_mels // 2:, :]
    return np.mean(top_half) # average energy of top half frequencies

sample_image_path = sample_images.iloc[0]['Image']
high_freq_energy = intgrl_img(sample_image_path)
overall_energy = mean_img(sample_image_path)

normal_train_df = train_df[train_df['Label'] == 0]

normal_features_ccv1_b1 = normal_train_df['Image'].apply(intgrl_img).values

threshold_mean_ccv1_b1 = np.mean(normal_features_ccv1_b1)
threshold_ccv1_b1 = threshold_mean_ccv1_b1 * 1.05

test_features_ccv1_b1 = test_df['Image'].apply(intgrl_img).values
test_true_ccv1_b1 = test_df['Label'].values

test_pred_ccv1_b1 = (test_features_ccv1_b1 > threshold_ccv1_b1).astype(int)

rec_t_ccv1_b1 = recall_score(test_true_ccv1_b1, test_pred_ccv1_b1, pos_label=1) * 100
prec_t_ccv1_b1 = precision_score(test_true_ccv1_b1, test_pred_ccv1_b1, pos_label=1) * 100
f1_t_ccv1_b1 = f1_score(test_true_ccv1_b1, test_pred_ccv1_b1, pos_label=1) * 100

print(f"\nAbnormal Recall: {rec_t_ccv1_b1:.1f}% \
    Abnormal Precision: {prec_t_ccv1_b1:.1f}% \
    Abnormal F1: {f1_t_ccv1_b1:.1f}%\n")
print(classification_report(test_true_ccv1_b1, test_pred_ccv1_b1, target_names=['Normal (0)', 'Abnormal (1)']))

cm = confusion_matrix(test_true_ccv1_b1, test_pred_ccv1_b1)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])
vis.plot(cmap="Grays", values_format='d', colorbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.yticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIG_CCV_DIR, 'ccv1_b1_test_cm.png'), dpi=300, bbox_inches='tight')

val_features_ccv1_b1 = val_df['Image'].apply(intgrl_img).values
val_true_ccv1_b1 = val_df['Label'].values

val_pred_ccv1_b1 = (val_features_ccv1_b1 > threshold_ccv1_b1).astype(int)

rec_v_ccv1_b1 = recall_score(val_true_ccv1_b1, val_pred_ccv1_b1, pos_label=1) * 100
prec_v_ccv1_b1 = precision_score(val_true_ccv1_b1, val_pred_ccv1_b1, pos_label=1) * 100
f1_v_ccv1_b1 = f1_score(val_true_ccv1_b1, val_pred_ccv1_b1, pos_label=1) * 100

print(f"\nAbnormal Recall: {rec_v_ccv1_b1:.1f}% \
    Abnormal Precision: {prec_v_ccv1_b1:.1f}% \
    Abnormal F1: {f1_v_ccv1_b1:.1f}%\n")
print(classification_report(val_true_ccv1_b1, val_pred_ccv1_b1, target_names=['Normal (0)', 'Abnormal (1)']))

cm = confusion_matrix(val_true_ccv1_b1, val_pred_ccv1_b1)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])
vis.plot(cmap="Grays", values_format='d', colorbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.yticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIG_CCV_DIR, 'ccv1_b1_val_cm.png'), dpi=300, bbox_inches='tight')