
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV

def rfecv(n_folds, add_data_scaled, data_scaled, add_features):
    # RFE feature selection with linear regression estimator
    lin_reg = LinearRegression()
    rfe = RFECV(estimator=lin_reg, cv=n_folds)
    rfe.fit(add_data_scaled, data_scaled.ravel())

    selected_features = np.array(add_features)[rfe.support_]
    # print(selected_features)

    return selected_features
