
import numpy as np
from sklearn.linear_model import LassoCV

def lassocv(n_folds, add_data_scaled, data_scaled, add_features):
    # LassoCV feature selection with cross-validation
    lassocv = LassoCV(cv=n_folds)
    lassocv.fit(add_data_scaled, data_scaled.ravel())

    selected_features = np.array(add_features)[lassocv.coef_ != 0]
    # print(selected_features)

    return selected_features
