from setup import *
from preprocess import test_df, val_df
from ccv_integral import normal_train_df
from explore import sample_images, vectorize_spec

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             recall_score,
                             precision_score,
                             f1_score)
from scipy.ndimage import sobel
import warnings
warnings.filterwarnings('ignore')

def edges_img(image_path):
    """
    Takes an image path, applies Sobel edge detection, and returns image with edges highlighted.
    """
    try:
        img = Image.open(image_path).convert('L')
        spec_array = np.array(img, dtype=np.float32)
    except Exception:
        return None 

    edge_detection = np.abs(sobel(spec_array, axis=1))
    
    return edge_detection

def count_edges(spec_array, n_mels=128):
    """Extracts the sum of vertical edges in the high-frequency half."""
    if spec_array is None: 
        return 0.0
    
    edge_detection = np.abs(sobel(spec_array, axis=1))
    
    high_freq_edges = edge_detection[n_mels // 2:, :]
    
    return np.sum(high_freq_edges)

img1 = sample_images.iloc[0]['Image']
img2 = sample_images.iloc[5]['Image']

edge_img1 = edges_img(img1)
edge_img2 = edges_img(img2)

edge_img1_count = count_edges(vectorize_spec(img1))
edge_img2_count = count_edges(vectorize_spec(img2))

fig, axes = plt.subplots(2, 2, figsize=(9,9))

axes[0, 0].imshow(plt.imread(img1), cmap='gray')
axes[0, 0].set_title(f'Sample A {"(Abnormal)" if sample_images.iloc[0]["Label"] == 1 else "(Normal)"}')
axes[0, 0].axis('off')

axes[0, 1].imshow(edge_img1, cmap='inferno')
axes[0, 1].set_title(f'Sample A Edges (count: {edge_img1_count:.0f})')
axes[0, 1].axis('off')

axes[1, 0].imshow(plt.imread(img2), cmap='gray')
axes[1, 0].set_title(f'Sample F {"(Abnormal)" if sample_images.iloc[5]["Label"] == 1 else "(Normal)"}')
axes[1, 0].axis('off')

axes[1, 1].imshow(edge_img2, cmap='inferno')
axes[1, 1].set_title(f'Sample F Edges (count: {edge_img2_count:.0f})')
axes[1, 1].axis('off')

plt.suptitle('Sobel Edge Detection on Sample Spectrograms', fontsize=16, y=0.92)
plt.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.savefig(os.path.join(FIG_CCV_DIR, 'sample_sobel_edges.png'), dpi=300, bbox_inches='tight')

normal_features_ccv1_b2 = normal_train_df['Image'].apply(vectorize_spec).apply(count_edges).values

threshold_mean_ccv1_b2 = np.mean(normal_features_ccv1_b2)
threshold_ccv1_b2 = threshold_mean_ccv1_b2 * 1.05

test_features_ccv1_b2 = test_df['Image'].apply(vectorize_spec).apply(count_edges).values
test_true_ccv1_b2 = test_df['Label'].values

test_pred_ccv1_b2 = (test_features_ccv1_b2 > threshold_ccv1_b2).astype(int)

rec_t_ccv1_b2 = recall_score(test_true_ccv1_b2, test_pred_ccv1_b2, pos_label=1) * 100
prec_t_ccv1_b2 = precision_score(test_true_ccv1_b2, test_pred_ccv1_b2, pos_label=1) * 100
f1_t_ccv1_b2 = f1_score(test_true_ccv1_b2, test_pred_ccv1_b2, pos_label=1) * 100

print(f"\nAbnormal Recall: {rec_t_ccv1_b2:.1f}% \
    Abnormal Precision: {prec_t_ccv1_b2:.1f}% \
    Abnormal F1: {f1_t_ccv1_b2:.1f}%\n")
print(classification_report(test_true_ccv1_b2, test_pred_ccv1_b2, target_names=['Normal (0)', 'Abnormal (1)']))

cm = confusion_matrix(test_true_ccv1_b2, test_pred_ccv1_b2)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])
vis.plot(cmap="Grays", values_format='d', colorbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.yticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIG_CCV_DIR, 'ccv1_b2_test_cm.png'), dpi=300, bbox_inches='tight')

val_features_ccv1_b2 = val_df['Image'].apply(vectorize_spec).apply(count_edges).values
val_true_ccv1_b2 = val_df['Label'].values

val_pred_ccv1_b2 = (val_features_ccv1_b2 > threshold_ccv1_b2).astype(int)

rec_v_ccv1_b2 = recall_score(val_true_ccv1_b2, val_pred_ccv1_b2, pos_label=1) * 100
prec_v_ccv1_b2 = precision_score(val_true_ccv1_b2, val_pred_ccv1_b2, pos_label=1) * 100
f1_v_ccv1_b2 = f1_score(val_true_ccv1_b2, val_pred_ccv1_b2, pos_label=1) * 100

print(f"\nAbnormal Recall: {rec_v_ccv1_b2:.1f}% \
    Abnormal Precision: {prec_v_ccv1_b2:.1f}% \
    Abnormal F1: {f1_v_ccv1_b2:.1f}%\n")
print(classification_report(val_true_ccv1_b2, val_pred_ccv1_b2, target_names=['Normal (0)', 'Abnormal (1)']))

cm = confusion_matrix(val_true_ccv1_b2, val_pred_ccv1_b2)
vis = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Abnormal (1)'])
vis.plot(cmap="Grays", values_format='d', colorbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.yticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIG_CCV_DIR, 'ccv1_b2_val_cm.png'), dpi=300, bbox_inches='tight')