from setup import *
from ccv_integral import *
from ccv_edges import *
from ccv_stat import *
from ccv_hog import *
from ccv_lbp import *

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import fbeta_score
import warnings
warnings.filterwarnings('ignore')


fbeta_t_ccv1_b1 = fbeta_score(test_true_ccv1_b1, test_pred_ccv1_b1, beta=2, pos_label=1) * 100
fbeta_t_ccv1_b2 = fbeta_score(test_true_ccv1_b2, test_pred_ccv1_b2, beta=2, pos_label=1) * 100

fbeta_v_ccv1_b1 = fbeta_score(val_true_ccv1_b1, val_pred_ccv1_b1, beta=2, pos_label=1) * 100
fbeta_v_ccv1_b2 = fbeta_score(val_true_ccv1_b2, val_pred_ccv1_b2, beta=2, pos_label=1) * 100

fbeta_metrics = pd.DataFrame({
    'Model': ['CCV1-B1', 'CCV1-B2'],
    'Test Set': [fbeta_t_ccv1_b1, fbeta_t_ccv1_b2],
    'Val Set': [fbeta_v_ccv1_b1, fbeta_v_ccv1_b2]
})  

metrics_ccv1 = pd.DataFrame({
    'Model': ['CCV1-B1', 'CCV1-B2'],
    'Test Recall': [rec_t_ccv1_b1, rec_t_ccv1_b2],
    'Test Precision': [prec_t_ccv1_b1, prec_t_ccv1_b2],
    'Test F1': [f1_t_ccv1_b1, f1_t_ccv1_b2],
    'Val Recall': [rec_v_ccv1_b1, rec_v_ccv1_b2],
    'Val Precision': [prec_v_ccv1_b1, prec_v_ccv1_b2],
    'Val F1': [f1_v_ccv1_b1, f1_v_ccv1_b2]
})

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

metrics_melted = metrics_ccv1.melt(id_vars='Model', var_name='Metric', value_name='Value')
sns.barplot(data=metrics_melted, x='Metric', y='Value', hue='Model', palette='Set2', edgecolor='white', ax=axes[0])
axes[0].set_title('Filter-Based Models Performance')
axes[0].set_ylabel('Performance (%)')
axes[0].set_xlabel('')
axes[0].set_ylim(0, 100)
axes[0].tick_params(axis='x', rotation=45)
model_names = {
    'CCV1-B1': 'CCV1-B1 (Integral Image Model)',
    'CCV1-B2': 'CCV1-B2 (Edge Detection Model)'
}
handles, labels = axes[0].get_legend_handles_labels()
new_labels = [model_names.get(label, label) for label in labels]
axes[0].legend(handles, new_labels, title='Model', loc='upper right')

fbeta_melted = fbeta_metrics.melt(id_vars='Model', var_name='Dataset', value_name='F2 Score')
sns.barplot(data=fbeta_melted, x='Model', y='F2 Score', hue='Dataset', palette='Accent', edgecolor='white', ax=axes[1])
axes[1].set_title('Filter-Based Models F-Beta Scores')
axes[1].set_ylabel('F-Beta (%)')
axes[1].set_xlabel('')
axes[1].set_ylim(0, 100)
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Dataset', loc='upper right')

plt.suptitle('CCV1 Models Evaluation', fontsize=18, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_CCV_DIR, 'ccv1_performance.png'), dpi=300, bbox_inches='tight')

fbeta_t_ccv2_b1 = fbeta_score(y_test_ccv2_b1_aligned, y_pred_test_ccv2_b1, beta=2, pos_label=1) * 100
fbeta_t_ccv2_b2 = fbeta_score(y_test_ccv2_b2_aligned, y_pred_test_ccv2_b2, beta=2, pos_label=1) * 100
fbeta_t_ccv2_b3 = fbeta_score(y_test_ccv2_b3_aligned, y_pred_test_ccv2_b3, beta=2, pos_label=1) * 100

fbeta_v_ccv2_b1 = fbeta_score(y_val_ccv2_b1_aligned, y_pred_val_ccv2_b1, beta=2, pos_label=1) * 100
fbeta_v_ccv2_b2 = fbeta_score(y_val_ccv2_b2_aligned, y_pred_val_ccv2_b2, beta=2, pos_label=1) * 100
fbeta_v_ccv2_b3 = fbeta_score(y_val_ccv2_b3_aligned, y_pred_val_ccv2_b3, beta=2, pos_label=1) * 100

fbeta_metrics = pd.DataFrame({
    'Model': ['CCV2-B1', 'CCV2-B2', 'CCV2-B3'],
    'Test Set': [fbeta_t_ccv2_b1, fbeta_t_ccv2_b2, fbeta_t_ccv2_b3],
    'Val Set': [fbeta_v_ccv2_b1, fbeta_v_ccv2_b2, fbeta_v_ccv2_b3]
})

metrics_ccv2 = pd.DataFrame({
    'Model': ['CCV2-B1', 'CCV2-B2', 'CCV2-B3'],
    'Test Recall': [rec_t_ccv2_b1, rec_t_ccv2_b2, rec_t_ccv2_b3],
    'Test Precision': [prec_t_ccv2_b1, prec_t_ccv2_b2, prec_t_ccv2_b3],
    'Test F1': [f1_t_ccv2_b1, f1_t_ccv2_b2, f1_t_ccv2_b3],
    'Val Recall': [rec_v_ccv2_b1, rec_v_ccv2_b2, rec_v_ccv2_b3],
    'Val Precision': [prec_v_ccv2_b1, prec_v_ccv2_b2, prec_v_ccv2_b3],
    'Val F1': [f1_v_ccv2_b1, f1_v_ccv2_b2, f1_v_ccv2_b3]
})

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

metrics_melted = metrics_ccv2.melt(id_vars='Model', var_name='Metric', value_name='Value')
sns.barplot(data=metrics_melted, x='Metric', y='Value', hue='Model', palette='Set2', edgecolor='white', ax=axes[0])
axes[0].set_title('Feature-Based Models Performance')
axes[0].set_ylabel('Performance (%)')
axes[0].set_xlabel('')
axes[0].set_ylim(0, 100)
axes[0].tick_params(axis='x', rotation=45)
model_names = {
    'CCV2-B1': 'CCV2-B1 (Stats)',
    'CCV2-B2': 'CCV2-B2 (HOG)',
    'CCV2-B3': 'CCV2-B3 (LBP)'
}
handles, labels = axes[0].get_legend_handles_labels()
new_labels = [model_names.get(label, label) for label in labels]
axes[0].legend(handles, new_labels, title='Model', loc='upper right')

fbeta_melted = fbeta_metrics.melt(id_vars='Model', var_name='Dataset', value_name='F2 Score')
sns.barplot(data=fbeta_melted, x='Model', y='F2 Score', hue='Dataset', palette='Accent', edgecolor='white', ax=axes[1])
axes[1].set_title('Feature-Based Models F-Beta Scores')
axes[1].set_ylabel('F-Beta (%)')
axes[1].set_xlabel('')
axes[1].set_ylim(0, 100)
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Dataset', loc='upper right')

plt.suptitle('CCV2 Models Evaluation', fontsize=18, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_CCV_DIR, 'ccv2_performance.png'), dpi=300, bbox_inches='tight')

fbeta_metrics = pd.DataFrame({
    'Model': ['CCV1-B1', 'CCV1-B2', 'CCV2-B1', 'CCV2-B2', 'CCV2-B3'],
    'Test Set': [fbeta_t_ccv1_b1, fbeta_t_ccv1_b2, fbeta_t_ccv2_b1, fbeta_t_ccv2_b2, fbeta_t_ccv2_b3],
    'Val Set': [fbeta_v_ccv1_b1, fbeta_v_ccv1_b2, fbeta_v_ccv2_b1, fbeta_v_ccv2_b2, fbeta_v_ccv2_b3]
})

plt.figure(figsize=(10,6))

fbeta_melted = fbeta_metrics.melt(id_vars='Model', var_name='Dataset', value_name='F2 Score')
sns.barplot(data=fbeta_melted, x='Model', y='F2 Score', hue='Dataset', palette='Accent', edgecolor='white')

plt.title('All Models F-Beta Scores')
plt.ylabel('F-Beta (%)')
plt.xlabel('')
plt.ylim(0, 100)
plt.tick_params(axis='x', rotation=45)
plt.legend(title='Dataset', loc='upper right')
plt.tight_layout()