import sys
from .rfecv import rfecv
from .lassocv import lassocv

def feature_engineering(feature_method, n_fold, add_data_scaled, data_scaled, add_feature):
  if feature_method == 'rfecv':
    selected_features = rfecv(n_fold, add_data_scaled, data_scaled, add_feature)

  elif feature_method == 'lassocv':
    selected_features = lassocv(n_fold, add_data_scaled, data_scaled, add_feature)

  else:
    print(f'feature_method not found: {feature_method}')
    sys.exit(0)

  return selected_features