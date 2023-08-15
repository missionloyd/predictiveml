import os, warnings, logging
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2', '3'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
