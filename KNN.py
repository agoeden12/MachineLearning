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

    # Clear the classified data and verifications results
    def clear(self):
        self.classification = []
        self.verification_results = {'pass': [], 'fail': []}

    # For each image row in the test data calculate the distance to every training set and select the k nearest neighbors
    def classify(self, data: np.ndarray, k):
        for data_row in data:

            # Create an empty distance array
            distances = []
            for (index, (label, values)) in enumerate(self.training_set):

                # Calculate distance to every training data and save that label, distance, and index (for debugging)
                distances.append((label, norm(data_row - values), index))

            # Sort the distances
            distances.sort(key=lambda dist: dist[1])

            # Get the k smallest distances and collect their labels
            neighbors = list(zip(*distances[:k]))[0]

            # Select the mode from the nearest labels
            mode = int(stats.mode(neighbors, keepdims=False)[0])
            self.classification.append(mode)

    # Assign the ground truth "training" labels and values in an easily parseable array
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
    
    # Create results array to store the x and y axis for graphing results
    results = [[],[]]

    # Loop through every odd k between 1 and 50
    for i in range(1,50,2):
        knn_model.classify(testing_data, i)
        knn_model.verify(testing_labels)
    
        # Add the result to the results array
        results[0].append(i)
        results[1].append(knn_model.accuracy)

        knn_model.print_results()
        knn_model.clear()

    # Graph the accuracy against the k values
    plt.plot(results[0], results[1])
    plt.show()
