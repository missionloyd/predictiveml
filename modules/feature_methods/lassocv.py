
import numpy as np
from sklearn.linear_model import LassoCV

def lassocv(n_fold, add_data_scaled, data_scaled, add_feature):
    # LassoCV feature selection with cross-validation
    lassocv = LassoCV(cv=n_fold)
    lassocv.fit(add_data_scaled, data_scaled.ravel())

    selected_features = np.array(add_feature)[lassocv.coef_ != 0]
    # print(selected_features)

    return selected_features
