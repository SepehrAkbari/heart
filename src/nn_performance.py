from setup import *
from nn_eval import *

import seaborn as sns

metrics_DL = pd.DataFrame({
    'Model': ['BaselineFNN', 'EnhancedFNN', 'BaselineCNN', 'MediumCNN', 'EnhancedCNN', 'seCNN', 'DFFN', 'AugmentedCNN'],
    'Test Recall': [rec_BaselineFNN * 100, rec_EnhancedFNN * 100, rec_BaselineCNN * 100, rec_MediumCNN * 100, rec_EnhancedCNN * 100, rec_seCNN * 100, rec_DFFN * 100, rec_AugmentedCNN * 100],
    'Test Precision': [prec_BaselineFNN * 100, prec_EnhancedFNN * 100, prec_BaselineCNN * 100, prec_MediumCNN * 100, prec_EnhancedCNN * 100, prec_seCNN * 100, prec_DFFN * 100, prec_AugmentedCNN * 100],
    'Test F1': [f1_BaselineFNN * 100, f1_EnhancedFNN * 100, f1_BaselineCNN * 100, f1_MediumCNN * 100, f1_EnhancedCNN * 100, f1_seCNN * 100, f1_DFFN * 100, f1_AugmentedCNN * 100]
})

metrics_melted = metrics_DL.melt(id_vars='Model', var_name='Metric', value_name='Value')
plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_melted, x='Model', y='Value', hue='Metric', palette='Accent')
plt.title('Deep Learning Model Performance Metrics')
plt.ylabel('Performance (%)')
plt.ylim(0, 100)
plt.legend(title='Metric', loc='upper right')
plt.xlabel('')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DL_DIR, 'DL_metric_performance.png'), dpi=300, bbox_inches='tight')

fbeta_metrics = pd.DataFrame({
    'Model': ['BaselineFNN', 'EnhancedFNN', 'BaselineCNN', 'MediumCNN', 'EnhancedCNN', 'seCNN', 'DFFN', 'AugmentedCNN'],
    'Test Set': [f2_BaselineFNN * 100, f2_EnhancedFNN * 100, f2_BaselineCNN * 100, f2_MediumCNN * 100, f2_EnhancedCNN * 100, f2_seCNN * 100, f2_DFFN * 100, f2_AugmentedCNN * 100]
})

plt.figure(figsize=(10, 6))
sns.barplot(data=fbeta_metrics, x='Model', y='Test Set', palette='Set3')
plt.title('Deep Learning Model F-Beta Scores')
plt.ylabel('F-Beta (%)')
plt.ylim(0, 100)
plt.xlabel('')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DL_DIR, 'DL_fbeta_performance.png'), dpi=300, bbox_inches='tight')


metrics_TL = pd.DataFrame({
    'Model': ['BaselineResnetTransfer', 'EnhancedResnetTransfer'],
    'Test Recall': [rec_BaselineResnetTransfer * 100, rec_EnhancedResnetTransfer * 100],
    'Test Precision': [prec_BaselineResnetTransfer * 100, prec_EnhancedResnetTransfer * 100],
    'Test F1': [f1_BaselineResnetTransfer * 100, f1_EnhancedResnetTransfer * 100]
})

fbeta_metrics = pd.DataFrame({
    'Model': ['BaselineResnetTransfer', 'EnhancedResnetTransfer'],
    'Test Set': [f2_BaselineResnetTransfer * 100, f2_EnhancedResnetTransfer * 100]
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

metrics_melted = metrics_TL.melt(id_vars='Model', var_name='Metric', value_name='Value')
sns.barplot(data=metrics_melted, x='Model', y='Value', hue='Metric', palette='Set3', ax=axes[0])
axes[0].set_title('Transfer Learning Model Performance Metrics')
axes[0].set_ylabel('Performance (%)')
axes[0].set_ylim(0, 100)
axes[0].legend(title='Metric', loc='upper right')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x')

sns.barplot(data=fbeta_metrics, x='Model', y='Test Set', palette='Accent', ax=axes[1])
axes[1].set_title('Transfer Learning Model F-Beta Scores')
axes[1].set_ylabel('F-Beta (%)')
axes[1].set_ylim(0, 100)
axes[1].set_xlabel('')
axes[1].tick_params(axis='x')

plt.suptitle('Transfer Learning Model Evaluation', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_TL_DIR, 'TL_performance.png'), dpi=300, bbox_inches='tight')