from setup import *
from preprocess import X_train, y_train, X_test, y_test, validation_df

from PIL import Image
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

def get_stats(image_path, n_mels=128):
    """Extracts statistical features from a spectrogram."""
    try:
        img = Image.open(image_path).convert('L')
        spec_array = np.array(img, dtype=np.float32)
        
        if spec_array is None or spec_array.size == 0:
            return None
        
        mean_intensity = np.mean(spec_array)
        std_intensity = np.std(spec_array)
        skewness = skew(spec_array.flatten())
        kurt = kurtosis(spec_array.flatten())
        
        high_freq_region = spec_array[n_mels // 2:, :]
        hfe = np.mean(high_freq_region)
        
        return [mean_intensity, std_intensity, skewness, kurt, hfe]
    
    except Exception:
        return None
    
X_train_ccv2_b1_feats = np.array([get_stats(p) for p in X_train.values if get_stats(p) is not None])
y_train_ccv2_b1_aligned = y_train.loc[X_train.index[~pd.Series([get_stats(p) for p in X_train.values]).isna().values]]

X_test_ccv2_b1_feats = np.array([get_stats(p) for p in X_test.values if get_stats(p) is not None])
y_test_ccv2_b1_aligned = y_test.loc[X_test.index[~pd.Series([get_stats(p) for p in X_test.values]).isna().values]]

X_val_ccv2_b1_feats = np.array([get_stats(p) for p in validation_df['Image'] if get_stats(p) is not None])
y_val_ccv2_b1_aligned = validation_df.loc[validation_df.index[~pd.Series([get_stats(p) for p in validation_df['Image']]).isna().values]]['Label'].values

scaler_ccv2_b1 = StandardScaler()

X_train_ccv2_b1_scaled = scaler_ccv2_b1.fit_transform(X_train_ccv2_b1_feats)
X_test_ccv2_b1_scaled = scaler_ccv2_b1.transform(X_test_ccv2_b1_feats)
X_val_ccv2_b1_scaled = scaler_ccv2_b1.transform(X_val_ccv2_b1_feats)

svm_ccv2_b1 = SVC(kernel='rbf', C=10, gamma='scale', random_state=325, class_weight='balanced')

svm_ccv2_b1.fit(X_train_ccv2_b1_scaled, y_train_ccv2_b1_aligned)
y_pred_test_ccv2_b1 = svm_ccv2_b1.predict(X_test_ccv2_b1_scaled)
y_pred_val_ccv2_b1 = svm_ccv2_b1.predict(X_val_ccv2_b1_scaled)

rec_t_ccv2_b1 = recall_score(y_test_ccv2_b1_aligned, y_pred_test_ccv2_b1, pos_label=1) * 100
prec_t_ccv2_b1 = precision_score(y_test_ccv2_b1_aligned, y_pred_test_ccv2_b1, pos_label=1) * 100
f1_t_ccv2_b1 = f1_score(y_test_ccv2_b1_aligned, y_pred_test_ccv2_b1, pos_label=1) * 100

print(f"\nAbnormal Recall: {rec_t_ccv2_b1:.1f}% \
    Abnormal Precision: {prec_t_ccv2_b1:.1f}% \
    Abnormal F1: {f1_t_ccv2_b1:.1f}%\n")
print(classification_report(y_test_ccv2_b1_aligned, y_pred_test_ccv2_b1, target_names=['Normal (0)', 'Abnormal (1)']))

cm = confusion_matrix(y_test_ccv2_b1_aligned, y_pred_test_ccv2_b1)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])
vis.plot(cmap="Grays", values_format='d', colorbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.yticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIG_CCV_DIR, 'ccv2_b1_test_cm.png'), dpi=300, bbox_inches='tight')

rec_v_ccv2_b1 = recall_score(y_val_ccv2_b1_aligned, y_pred_val_ccv2_b1, pos_label=1) * 100
prec_v_ccv2_b1 = precision_score(y_val_ccv2_b1_aligned, y_pred_val_ccv2_b1, pos_label=1) * 100
f1_v_ccv2_b1 = f1_score(y_val_ccv2_b1_aligned, y_pred_val_ccv2_b1, pos_label=1) * 100

print(f"\nAbnormal Recall: {rec_v_ccv2_b1:.1f}% \
    Abnormal Precision: {prec_v_ccv2_b1:.1f}% \
    Abnormal F1: {f1_v_ccv2_b1:.1f}%\n")
print(classification_report(y_val_ccv2_b1_aligned, y_pred_val_ccv2_b1, target_names=['Normal (0)', 'Abnormal (1)']))

cm = confusion_matrix(y_val_ccv2_b1_aligned, y_pred_val_ccv2_b1)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])
vis.plot(cmap="Grays", values_format='d', colorbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.yticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIG_CCV_DIR, 'ccv2_b1_val_cm.png'), dpi=300, bbox_inches='tight')