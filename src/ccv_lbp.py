from setup import *
from explore import sample_images
from preprocess import X_train, y_train, X_test, y_test, validation_df

from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

def show_lbp(image_path, n_points=10, radius=1):
    """Returns LBP image and histogram for a given spectrogram."""
    try:
        img = Image.open(image_path).convert('L')
        spec_array = np.array(img, dtype=np.uint8)
        
        lbp = local_binary_pattern(spec_array, P=n_points, R=radius, method='uniform')
        
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, n_points + 2),
                                 range=(0, n_points + 1))
        
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        return lbp, hist
    
    except:
        return None, None
    
img1 = sample_images.iloc[0]['Image']
img2 = sample_images.iloc[5]['Image']

lbp_image1, lbp_hist1 = show_lbp(img1)
lbp_image2, lbp_hist2 = show_lbp(img2)

fig, axes = plt.subplots(2, 3, figsize=(12,9))

axes[0, 0].imshow(plt.imread(img1), cmap='magma')
axes[0, 0].set_title(f'Sample A {"(Abnormal)" if sample_images.iloc[0]["Label"] == 1 else "(Normal)"}')
axes[0, 0].axis('off')
axes[0, 1].imshow(lbp_image1, cmap='plasma')
axes[0, 1].set_title('LBP Image')
axes[0, 1].axis('off')
axes[0, 2].hist(lbp_image1.ravel(), bins=np.arange(0, 12), density=True, color='violet', alpha=0.4, edgecolor='white')
axes[0, 2].plot(lbp_hist1, color='purple', linewidth=2)
axes[0, 2].set_title('LBP Histogram')
axes[0, 2].set_xlabel('LBP Pattern')
axes[0, 2].set_ylabel('Normalized Frequency')

axes[1, 0].imshow(plt.imread(img2), cmap='magma')
axes[1, 0].set_title(f'Sample F {"(Abnormal)" if sample_images.iloc[5]["Label"] == 1 else "(Normal)"}')
axes[1, 0].axis('off')
axes[1, 1].imshow(lbp_image2, cmap='plasma')
axes[1, 1].set_title('LBP Image')
axes[1, 1].axis('off')
axes[1, 2].hist(lbp_image2.ravel(), bins=np.arange(0, 12), density=True, color='violet', alpha=0.4, edgecolor='white')
axes[1, 2].plot(lbp_hist2, color='purple', linewidth=2)
axes[1, 2].set_title('LBP Histogram')
axes[1, 2].set_xlabel('LBP Pattern')
axes[1, 2].set_ylabel('Normalized Frequency')

plt.suptitle('Local Binary Patterns on Sample Spectrograms', fontsize=16, y=0.92)
plt.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.savefig(os.path.join(FIG_CCV_DIR, 'sample_lbp_features.png'), dpi=300, bbox_inches='tight')

def get_lbp(image_path, n_points=10, radius=1):
    """Applies LBP, and computes the LBP histogram."""
    try:
        img = Image.open(image_path).convert('L')
        spec_array = np.array(img, dtype=np.uint8)
        
        lbp = local_binary_pattern(spec_array, P=n_points, R=radius, method='uniform')
        
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, n_points + 2),
                                 range=(0, n_points + 1))
        
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        return hist
    
    except:
        return None
    
X_train_ccv2_b3_feats = [get_lbp(p) for p in X_train.values]
X_train_ccv2_b3_feats = np.array([f for f in X_train_ccv2_b3_feats if f is not None])
y_train_ccv2_b3_aligned = y_train.loc[X_train.index[:len(X_train_ccv2_b3_feats)]].values

X_test_ccv2_b3_feats = [get_lbp(p) for p in X_test.values]
X_test_ccv2_b3_feats = np.array([f for f in X_test_ccv2_b3_feats if f is not None])
y_test_ccv2_b3_aligned = y_test.loc[X_test.index[:len(X_test_ccv2_b3_feats)]].values

X_val_ccv2_b3_feats = [get_lbp(p) for p in validation_df['Image'] if get_lbp(p) is not None]
X_val_ccv2_b3_feats = np.array([f for f in X_val_ccv2_b3_feats if f is not None])
y_val_ccv2_b3_aligned = validation_df['Label'].values[:len(X_val_ccv2_b3_feats)]

scaler_ccv2_b3 = StandardScaler()

X_train_ccv2_b3_scaled = scaler_ccv2_b3.fit_transform(X_train_ccv2_b3_feats)
X_test_ccv2_b3_scaled = scaler_ccv2_b3.transform(X_test_ccv2_b3_feats)
X_val_ccv2_b3_scaled = scaler_ccv2_b3.transform(X_val_ccv2_b3_feats)

knn_ccv2_b3 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='uniform')

knn_ccv2_b3.fit(X_train_ccv2_b3_scaled, y_train_ccv2_b3_aligned)
y_pred_test_ccv2_b3 = knn_ccv2_b3.predict(X_test_ccv2_b3_scaled)
y_pred_val_ccv2_b3 = knn_ccv2_b3.predict(X_val_ccv2_b3_scaled)

rec_t_ccv2_b3 = recall_score(y_test_ccv2_b3_aligned, y_pred_test_ccv2_b3, pos_label=1) * 100
prec_t_ccv2_b3 = precision_score(y_test_ccv2_b3_aligned, y_pred_test_ccv2_b3, pos_label=1) * 100
f1_t_ccv2_b3 = f1_score(y_test_ccv2_b3_aligned, y_pred_test_ccv2_b3, pos_label=1) * 100

print(f"\nAbnormal Recall: {rec_t_ccv2_b3:.1f}% \
    Abnormal Precision: {prec_t_ccv2_b3:.1f}% \
    Abnormal F1: {f1_t_ccv2_b3:.1f}%\n")
print(classification_report(y_test_ccv2_b3_aligned, y_pred_test_ccv2_b3, target_names=['Normal (0)', 'Abnormal (1)']))

cm = confusion_matrix(y_test_ccv2_b3_aligned, y_pred_test_ccv2_b3)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])
vis.plot(cmap="Grays", values_format='d', colorbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.yticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIG_CCV_DIR, 'ccv2_b3_test_cm.png'), dpi=300, bbox_inches='tight')

rec_v_ccv2_b3 = recall_score(y_val_ccv2_b3_aligned, y_pred_val_ccv2_b3, pos_label=1) * 100
prec_v_ccv2_b3 = precision_score(y_val_ccv2_b3_aligned, y_pred_val_ccv2_b3, pos_label=1) * 100
f1_v_ccv2_b3 = f1_score(y_val_ccv2_b3_aligned, y_pred_val_ccv2_b3, pos_label=1) * 100

print(f"\nAbnormal Recall: {rec_v_ccv2_b3:.1f}% \
    Abnormal Precision: {prec_v_ccv2_b3:.1f}% \
    Abnormal F1: {f1_v_ccv2_b3:.1f}%\n")
print(classification_report(y_val_ccv2_b3_aligned, y_pred_val_ccv2_b3, target_names=['Normal (0)', 'Abnormal (1)']))

cm = confusion_matrix(y_val_ccv2_b3_aligned, y_pred_val_ccv2_b3)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])
vis.plot(cmap="Grays", values_format='d', colorbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.yticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIG_CCV_DIR, 'ccv2_b3_val_cm.png'), dpi=300, bbox_inches='tight')