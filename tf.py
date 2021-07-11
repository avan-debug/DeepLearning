import numpy as np
import time
from test2.lr_utils import load_dataset
import matplotlib.pyplot as plt

if __name__ == '__main__':
     a = np.random.randn(2, 3, 4)
     b = a.reshape(2, -1)
     print(b)