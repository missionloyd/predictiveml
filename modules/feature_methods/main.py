import sys
from .rfecv import rfecv
from .lassocv import lassocv

def feature_engineering(feature_method, n_folds, add_data_scaled, data_scaled, add_features):
  if feature_method == 'rfecv':
    selected_features = rfecv(n_folds, add_data_scaled, data_scaled, add_features)

  elif feature_method == 'lassocv':
    selected_features = lassocv(n_folds, add_data_scaled, data_scaled, add_features)

  else:
    print(f'feature_method not found: {feature_method}')
    sys.exit(0)

  return selected_features