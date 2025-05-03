import sys
import tensorflow as tf
from tensorflow.python.keras.layers import GRU

print(f"Python路径: {sys.executable}")
print(f"TensorFlow版本: {tf.__version__}")
print(f"GRU层测试: {GRU(10).get_config()['name']}")