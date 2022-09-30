import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

testing_set = pd.read_csv('HW1/MNIST_training_HW1.csv')
testing_data = np.array(testing_set[testing_set.columns.difference(['label'])])
testing_label = np.array(testing_set['label'])

fig, ax = plt.subplots(nrows=10, ncols=10)

for index, value in enumerate(testing_data):
    if index == 100:
        break
    row, col = int(index / 10) if index > 9 else int((index / 10) % 10), int(index % 10) if index > 9 else index
    ax[row][col].imshow(value.reshape(28, -1), cmap='gray')
    ax[row][col].set_axis_off()
    ax[row][col].set_title(testing_label[index])


plt.show()
