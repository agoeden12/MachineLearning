import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy import stats
from matplotlib import pyplot as plt

class KNN():

    # Class variables to better manage data
    classification = []
    training_set = {}
    accuracy = None
    verification_results = {'pass': [], 'fail': []}

    def clear(self):
        self.classification = []
        self.verification_results = {'pass': [], 'fail': []}

    def classify(self, data: np.ndarray, k):
        for data_row in data:
            distances = []
            for (index, (label, values)) in enumerate(self.training_set):
                distances.append((label, norm(data_row - values), index))
            distances.sort(key=lambda dist: dist[1])

            neighbors = list(zip(*distances[:k]))[0]
            mode = int(stats.mode(neighbors, keepdims=False)[0])
            self.classification.append(mode)

    def fit(self, labels, values):
        self.training_set = list(zip(labels, values))

    # Calculate out how many tests passed of the classified data
    def verify(self, expected: np.ndarray):
        for index, value in enumerate(self.classification):
            self.verification_results['pass' if value == expected[index] else 'fail'].append(index)
        self.accuracy = len(self.verification_results['pass']) / len(self.classification)

    # Print results in a readable fashion
    def print_results(self):
        no_pass = len(self.verification_results['pass'])
        no_fail = len(self.verification_results['fail'])
        print(f'Accuracy: {self.accuracy} | Number Passed: {no_pass} | Number Failed {no_fail}')

if __name__ == '__main__':

    # Load the training sets and split into two arrays of labels and data
    training_set = pd.read_csv('HW2/MNIST_training_HW2.csv')
    training_labels = np.array(training_set['label'])
    training_data = np.array(training_set[training_set.columns.difference(['label'])])

    # Load the testing sets and split into two arrays of labels and data
    testing_set = pd.read_csv('HW2/MNIST_test_HW2.csv')
    testing_labels = np.array(testing_set['label'])
    testing_data = np.array(testing_set[testing_set.columns.difference(['label'])])

    knn_model = KNN()
    knn_model.fit(training_labels, training_data)

    results = [[],[]]
    for i in range(1,50,2):
        knn_model.classify(testing_data, i)
        knn_model.verify(testing_labels)
        results[0].append(i)
        results[1].append(knn_model.accuracy)

        knn_model.print_results()
        knn_model.clear()

    plt.plot(results[0], results[1])
    plt.show()
