import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class KNN():

    # Class variables to better manage data
    classification = []
    weights = []
    verification_results = {'pass': [], 'fail': []}

    # Use the theta weights to classify given data
    def classify(self, data: np.ndarray):
        pass

    # Calculate out how many tests passed of the classified data
    def verify(self, expected: np.ndarray):
        for index, value in enumerate(self.classification):
            self.verification_results['pass' if value == expected[index] else 'fail'].append(index)

    # Print results in a readable fashion
    def print_results(self):
        accuracy = len(self.verification_results['pass']) / len(self.classification)
        no_pass = len(self.verification_results['pass'])
        no_fail = len(self.verification_results['fail'])
        print(f'Accuracy: {accuracy} | Number Passed: {no_pass} | Number Failed {no_fail}')

if __name__ == '__main__':

    # Load the training sets and split into two arrays of labels and data
    training_set = pd.read_csv('HW2/MNIST_training_HW2.csv')
    training_labels = np.array(training_set['label'])
    training_data = np.array(training_set[training_set.columns.difference(['label'])])

    # Load the testing sets and split into two arrays of labels and data
    testing_set = pd.read_csv('HW2/MNIST_test_HW2.csv')
    testing_labels = np.array(testing_set['label'])
    testing_data = np.array(testing_set[testing_set.columns.difference(['label'])])
