# Fast-CMIM
Fast but exact implementation of CMIM feature selection algorithm

____

This is a fast implementation of the Conditional Mutual Information Maximisation (CMIM) feature selection algorithm. 
CMIM iteratively selects features by maximizing MI with a target variable, conditionned on previously selected features.

Speed: 100 features selected in ~180 000 ternary features in about 7 minutes.


- For the rationale of this algorithm and of the fast implementation, see [this webpage](http://www.ams.jhu.edu/~yqin/cvrg/Feature_Selection.htm#ch1_2)
- For the seminal article about this method, see :  [Fleuret 2004](http://www.idiap.ch/~fleuret/papers/fleuret-jmlr2004.pdf)


Note: this implementation is based on a different and slower (by orders of magnitude, say ~300 times) version of CMIM

See if interested [skfeature CMIM implementation](https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/information_theoretical_based/CMIM.py) which has a different objective function.