from setup import *
from explore import sample_images
from preprocess import X_train, y_train, X_test, y_test, validation_df

from PIL import Image
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

hog_orientations = 10
hog_pixels_per_cell = (16, 16)
hog_cells_per_block = (2, 2)

img1 = sample_images.iloc[0]['Image']
img2 = sample_images.iloc[5]['Image']

img1 = Image.open(img1).convert('L')
img2 = Image.open(img2).convert('L')

img_arr1 = np.array(img1, dtype=np.float32)
img_arr2 = np.array(img2, dtype=np.float32)

hog1, hog_image1 = hog(
    img_arr1, 
    orientations=hog_orientations,
    pixels_per_cell=hog_pixels_per_cell,
    cells_per_block=hog_cells_per_block,
    visualize=True
)

hog2, hog_image2 = hog(
    img_arr2, 
    orientations=hog_orientations,
    pixels_per_cell=hog_pixels_per_cell,
    cells_per_block=hog_cells_per_block,
    visualize=True
)

fig, axes = plt.subplots(2, 2, figsize=(9,9))
axes[0, 0].imshow(img_arr1, cmap='magma')
axes[0, 0].set_title(f'Sample A {"(Abnormal)" if sample_images.iloc[0]["Label"] == 1 else "(Normal)"}')
axes[0, 0].axis('off')

axes[0, 1].imshow(hog_image1, cmap='inferno')
axes[0, 1].set_title('Sample A HOG Features')
axes[0, 1].axis('off')

axes[1, 0].imshow(img_arr2, cmap='magma')
axes[1, 0].set_title(f'Sample F {"(Abnormal)" if sample_images.iloc[5]["Label"] == 1 else "(Normal)"}')
axes[1, 0].axis('off')

axes[1, 1].imshow(hog_image2, cmap='inferno')
axes[1, 1].set_title('Sample F HOG Features')
axes[1, 1].axis('off')

plt.suptitle('HOG Feature Extraction on Sample Spectrograms', fontsize=16, y=0.92)
plt.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.savefig(os.path.join(FIG_CCV_DIR, 'sample_hog_features.png'), dpi=300, bbox_inches='tight')

def get_hog(image_path):
    """Normalizes, and extracts HOG feature vector."""
    try:
        img = Image.open(image_path).convert('L')
        spec_array = np.array(img, dtype=np.float32)
        
        spec_array /= 255.0
        
        features = hog(
            spec_array, 
            orientations=10,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            transform_sqrt=True, # for noise reduction
            feature_vector=True
        )
        return features

    except:
        return None
    
X_train_ccv2_b2_feats = [get_hog(p) for p in X_train.values]
X_train_ccv2_b2_feats = np.array([f for f in X_train_ccv2_b2_feats if f is not None])
y_train_ccv2_b2_aligned = y_train.loc[X_train.index[:len(X_train_ccv2_b2_feats)]].values


X_test_ccv2_b2_feats = [get_hog(p) for p in X_test.values]
X_test_ccv2_b2_feats = np.array([f for f in X_test_ccv2_b2_feats if f is not None])
y_test_ccv2_b2_aligned = y_test.loc[X_test.index[:len(X_test_ccv2_b2_feats)]].values

X_val_ccv2_b2_feats = np.array([get_hog(p) for p in validation_df['Image'] if get_hog(p) is not None])
X_val_ccv2_b2_feats = np.array([f for f in X_val_ccv2_b2_feats if f is not None])
y_val_ccv2_b2_aligned = validation_df['Label'].values[:len(X_val_ccv2_b2_feats)]

scaler_ccv2_b2 = StandardScaler()
X_train_ccv2_b2_scaled = scaler_ccv2_b2.fit_transform(X_train_ccv2_b2_feats)
X_test_ccv2_b2_scaled = scaler_ccv2_b2.transform(X_test_ccv2_b2_feats)
X_val_ccv2_b2_scaled = scaler_ccv2_b2.transform(X_val_ccv2_b2_feats)

rf_ccv2_b2 = RandomForestClassifier(
    n_estimators=20,
    max_depth=10,
    random_state=325, 
    class_weight='balanced',
    n_jobs=-1
)

rf_ccv2_b2.fit(X_train_ccv2_b2_scaled, y_train_ccv2_b2_aligned)
y_pred_test_ccv2_b2 = rf_ccv2_b2.predict(X_test_ccv2_b2_scaled)
y_pred_val_ccv2_b2 = rf_ccv2_b2.predict(X_val_ccv2_b2_scaled)

rec_t_ccv2_b2 = recall_score(y_test_ccv2_b2_aligned, y_pred_test_ccv2_b2, pos_label=1) * 100
prec_t_ccv2_b2 = precision_score(y_test_ccv2_b2_aligned, y_pred_test_ccv2_b2, pos_label=1) * 100
f1_t_ccv2_b2 = f1_score(y_test_ccv2_b2_aligned, y_pred_test_ccv2_b2, pos_label=1) * 100

print(f"\nAbnormal Recall: {rec_t_ccv2_b2:.1f}% \
    Abnormal Precision: {prec_t_ccv2_b2:.1f}% \
    Abnormal F1: {f1_t_ccv2_b2:.1f}%\n")
print(classification_report(y_test_ccv2_b2_aligned, y_pred_test_ccv2_b2, target_names=['Normal (0)', 'Abnormal (1)']))

cm = confusion_matrix(y_test_ccv2_b2_aligned, y_pred_test_ccv2_b2)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])
vis.plot(cmap="Grays", values_format='d', colorbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.yticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIG_CCV_DIR, 'ccv2_b2_test_cm.png'), dpi=300, bbox_inches='tight')

rec_v_ccv2_b2 = recall_score(y_val_ccv2_b2_aligned, y_pred_val_ccv2_b2, pos_label=1) * 100
prec_v_ccv2_b2 = precision_score(y_val_ccv2_b2_aligned, y_pred_val_ccv2_b2, pos_label=1) * 100
f1_v_ccv2_b2 = f1_score(y_val_ccv2_b2_aligned, y_pred_val_ccv2_b2, pos_label=1) * 100

print(f"\nAbnormal Recall: {rec_v_ccv2_b2:.1f}% \
    Abnormal Precision: {prec_v_ccv2_b2:.1f}% \
    Abnormal F1: {f1_v_ccv2_b2:.1f}%\n")
print(classification_report(y_val_ccv2_b2_aligned, y_pred_val_ccv2_b2, target_names=['Normal (0)', 'Abnormal (1)']))

cm = confusion_matrix(y_val_ccv2_b2_aligned, y_pred_val_ccv2_b2)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])
vis.plot(cmap="Grays", values_format='d', colorbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.yticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIG_CCV_DIR, 'ccv2_b2_val_cm.png'), dpi=300, bbox_inches='tight')

