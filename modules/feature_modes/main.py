import sys
from .rfecv import rfecv
from .lassocv import lassocv

def feature_engineering(feature_mode, n_folds, add_data_scaled, data_scaled, add_features):
  if feature_mode == 'rfecv':
    selected_features = rfecv(n_folds, add_data_scaled, data_scaled, add_features)

  elif feature_mode == 'lassocv':
    selected_features = lassocv(n_folds, add_data_scaled, data_scaled, add_features)

  else:
    print(f'feature_mode not found: {feature_mode}')
    sys.exit(0)

  return selected_features