"""
This is a fast implementation of the Conditional Mutual Information Maximisation (CMIM) algorithm.
CMIM iteratively selects features by maximizing MI with a target variable, conditionned on previously selected
features.

Speed: 100 features selected in ~180 000 ternary features in about 7 minutes.

For the rationale of this algorithm and of the fast implementation, see:
http://www.ams.jhu.edu/~yqin/cvrg/Feature_Selection.htm#ch1_2
For the seminal article about this method, see : 
http://www.idiap.ch/~fleuret/papers/fleuret-jmlr2004.pdf

Note: this implementation is based on a different and slower (by orders of magnitude) version of CMIM
See if interested:
https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/information_theoretical_based/CMIM.py
The objective function is different, but output t1 here corresponds to jcmim output of skfeature cmim.
"""

import numpy as np
from utils import *

def fast_cmim(X, y, **kwargs):
    """
    This function implements the CMIM feature selection.
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete numpy array
    y: {numpy array}, shape (n_samples,)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    t1: {numpy array}, shape: (n_features,)
        minimal corresponding mutual information between selected features and response when 
        conditionned on a previously selected feature
    Reference
    ---------
    Fleuret 2004 - Fast Binary Feature Selection with Conditional Mutual Information
    http://www.idiap.ch/~fleuret/papers/fleuret-jmlr2004.pdf
    """

    n_samples, n_features = X.shape
    is_n_selected_features_specified = False

    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True
        F = np.nan * np.zeros(n_selected_features)
    else:
        F = np.nan * np.zeros(n_features)

    # t1
    t1 = np.zeros(n_features)
    
    # m is a counting indicator
    m = np.zeros(n_features) - 1
    
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)
    
    
    for k in range(n_features):
        ### uncomment to keep track
        # counter = int(np.sum(~np.isnan(F)))
        # if counter%5 == 0 or counter <= 1:
        #     print("F contains %s features"%(counter))
        if k == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F[0] = idx
            f_select = X[:, idx]

        if is_n_selected_features_specified:
            if np.sum(~np.isnan(F)) == n_selected_features:
                break

        sstar = -1000000 # start with really low value for best partial score sstar 
        for i in range(n_features):
            
            if i not in F:
                
                while (t1[i] > sstar) and (m[i]<k-1) :
                    m[i] = m[i] + 1
                    t1[i] = min(t1[i], cmidd(X[:,i], # feature i
                                             y,  # target
                                             X[:, int(F[int(m[i])])] # conditionned on selected features
                                            )
                               )
                if t1[i] > sstar:
                    sstar = t1[i]
                    F[k+1] = i
                    
    F = np.array(F[F>-100])
    F = F.astype(int)
    t1 = t1[F]
    return (F, t1)