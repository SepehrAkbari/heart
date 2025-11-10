from setup import *

import pandas as pd
import numpy as np

all_models = ['ccv1_b1', 'ccv1_b2',
              'ccv2_b1', 'ccv2_b2', 'ccv2_b3',
              'BaselineFNN', 'EnhancedFNN',
              'BaselineCNN', 'MediumCNN', 'EnhancedCNN',
              'seCNN', 'DFFN', 'AugmentedCNN',
              'BaselineResnetTransfer', 'EnhancedResnetTransfer',
              'Ensemble']

metrics = ['Recall 0', 'Precision 1', 'F1', 'F2']
def get_rec0(tn, fp, fn, tp):
    if (tp + fn) == 0:
        return 0.0
    return tp / (tp + fn)

def get_prec1(tn, fp, fn, tp):
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)

def get_f(tn, fp, fn, tp, beta):
    if (tp + fp) == 0:
        prec0 = 0.0
    else:
        prec0 = tp / (tp + fp)
        
    rec0 = get_rec0(tn, fp, fn, tp)
    
    if prec0 == 0 and rec0 == 0:
        return 0.0
        
    return ((1 + beta**2) * prec0 * rec0) / ((beta**2 * prec0) + rec0)

cm_data = {
    'ccv1_b1': np.array([[195, 239], [74, 39]]),
    'ccv1_b2': np.array([[402, 132], [30, 83]]),
    'ccv2_b1': np.array([[362, 172], [39, 74]]),
    'ccv2_b2': np.array([[495, 39], [51, 62]]),
    'ccv2_b3': np.array([[508, 26], [55, 58]]),
    'BaselineFNN': np.array([[341, 193], [6, 107]]),
    'EnhancedFNN': np.array([[434, 100], [18, 95]]),
    'BaselineCNN': np.array([[433, 101], [12, 101]]),
    'MediumCNN': np.array([[373, 161], [5, 108]]),
    'EnhancedCNN': np.array([[467, 67], [19, 94]]),
    'seCNN': np.array([[369, 165], [13, 100]]),
    'DFFN': np.array([[446, 88], [8, 105]]),
    'AugmentedCNN': np.array([[358, 176], [4, 109]]),
    'BaselineResnetTransfer': np.array([[445, 89], [19, 94]]),
    'EnhancedResnetTransfer': np.array([[453, 81], [9, 104]]),
    'Ensemble': np.array([[436, 98], [5, 108]])
}

results = []
for model_name, cm in cm_data.items():
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    r0 = get_rec0(tn, fp, fn, tp)
    p1 = get_prec1(tn, fp, fn, tp)
    f1 = get_f(tn, fp, fn, tp, beta = 1.0)
    f2 = get_f(tn, fp, fn, tp, beta = 2.0)
    
    results.append({
        'Model': model_name,
        'Recall 0': r0 * 100, 'Precision 1': p1 * 100,
        'F1': f1 * 100, 'F2': f2 * 100
    })

summary_df = pd.DataFrame(results).set_index('Model').round(2)
summary_df = summary_df.sort_values(by='F2', ascending=False)

