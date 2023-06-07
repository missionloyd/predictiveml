
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV

def rfecv(n_fold, add_data_scaled, data_scaled, add_feature):
    # RFE feature selection with linear regression estimator
    lin_reg = LinearRegression()
    rfe = RFECV(estimator=lin_reg, cv=n_fold)
    rfe.fit(add_data_scaled, data_scaled.ravel())

    selected_features = np.array(add_feature)[rfe.support_]
    # print(selected_features)

    return selected_features
