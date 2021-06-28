import numpy as np

def get_gt_attrib_ann(ann, X, hidden_activations='relu'):
    if hidden_activations == 'relu':
        out = (X @ ann.coefs_[0]) + ann.intercepts_[0]
        return np.where(out > 0, ann.coefs_[0], 0)