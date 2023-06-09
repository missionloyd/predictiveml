import sys
from .rfecv import rfecv
from .lassocv import lassocv
from modules.logging_methods.main import logger

def feature_engineering(feature_method, n_fold, add_data_scaled, data_scaled, add_feature):

  if n_fold > len(data_scaled):
    n_fold = len(data_scaled)

  if feature_method == 'rfecv':
    selected_features = rfecv(n_fold, add_data_scaled, data_scaled, add_feature)

  elif feature_method == 'lassocv':
    selected_features = lassocv(n_fold, add_data_scaled, data_scaled, add_feature)

  else:
    logger(f'feature_method not found: {feature_method}')
    sys.exit(0)

  return selected_features