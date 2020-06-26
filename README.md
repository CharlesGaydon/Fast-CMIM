# Fast-CMIM
A fast (but exact) implementation of CMIM feature selection algorithm

> This is a fast implementation of the Conditional Mutual Information Maximisation (CMIM) feature selection algorithm. 
CMIM iteratively selects features by maximizing Mutual Information (MI) with a target variable, conditionned on previously selected features.

Indicative speed: 100 features selected among ~180 000 three-modalities features in about 7 minutes.


___

References:

- For the rationale and pseudo-code of the fast CMIM implementation, see the [CVRG website > Feature Selection > CMIM Implementations > Fast Implementation](http://wiki.cvrgrid.org/index.php/FeatureSelection)
- For the seminal article about CMIM, see [Fleuret 2004 - Fast Binary Feature Selection with Conditional Mutual Information
](http://www.idiap.ch/~fleuret/papers/fleuret-jmlr2004.pdf).
- If interested in other implementations of CMIM, see [scikit-feature's CMIM implementation](https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/information_theoretical_based/CMIM.py) which has a different objective function. The skfeature implementation is based on a different and slower (by orders of magnitude, say ~300 times) version of CMIM
