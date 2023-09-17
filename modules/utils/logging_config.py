import os, warnings
import tensorflow as tf

def logging_config():
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2', '3'}
  tf.keras.utils.disable_interactive_logging()
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  return
