import numpy as np
import time
from test2.lr_utils import load_dataset
import matplotlib.pyplot as plt

if __name__ == '__main__':
    a = np.array([[1, 3, 4]])
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

    index = 24
    plt.imshow(train_set_x_orig[index])
    print("y={} is a {} picture".format(train_set_y_orig[:, index], classes[np.squeeze(train_set_y_orig[:, index])].decode("utf-8")))