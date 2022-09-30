import pandas as pd
import numpy as np
from numpy.linalg import pinv
from matplotlib import pyplot as plt

class LinearRegression():

    # Class variables to better manage data
    gradient_descent_costs = []
    classification = []
    verification_results = {'pass': [], 'fail': []}
    weights = []

    # Clear the local python memory because it wouldn't create a new instance
    def clear(self):
        self.gradient_descent_costs = []
        self.classification = []
        self.weights = []
        self.verification_results = {'pass': [], 'fail': []}

    # Use the theta weights to classify given data
    def classify(self, data: np.ndarray):
        print(f'Coefficients: {self.weights}')
        for value in data:
            self.classification.append(1 if (value * self.weights).sum() > 0.5 else 0)

    # Generate the weights using a normal equation
    def normal_equation(self, labels: np.ndarray, data: np.ndarray):
        inverse = pinv(np.matmul(data.T, data))
        prod = data.T * labels
        coeff = np.matmul(inverse,prod)
        self.weights = coeff.sum(axis=1)

    # Calculate the total cost of the given theta weight set
    def cost(self, labels, data):
        total_cost = 0

        # For each row calculate the cost against the expected value
        for index, values in enumerate(data):
            total_cost += (np.dot(self.weights, values) - labels[index])**2

        return (1 / 2 * len(labels)) * total_cost

    # m = training data
    # n = features
    def gradient_descent(self, step: float, iterations: int, labels: np.ndarray, data: np.ndarray):

        # Initialize local variables
        m = len(labels)
        columns = data.T
        rows = data

        thetas = np.zeros(len(columns))
        self.weights = np.zeros(len(columns))

        # Adjust weights for iterations
        for i in range(iterations):

            # Loop through each feature and adjust the temporary weights
            for col_index, col_value in enumerate(columns):
                theta_sum = 0

                # Sum up the total error of the specific feature in each column with respect to the current weight
                for row_index, row in enumerate(rows):
                    hypothesis = np.dot(self.weights, row)
                    theta_sum += (hypothesis - labels[row_index]) * col_value[row_index]

                # Update the temporary weight
                thetas[col_index] = self.weights[col_index] - ((step / m) * theta_sum)

            # Set the global weight simultaneously
            self.weights = thetas

            # Calculate cost and save for graphical display
            self.gradient_descent_costs.append(self.cost(labels, data))

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
    training_set = pd.read_csv('MNIST_training_HW1.csv')
    training_labels = np.array(training_set['label'])
    training_data = np.array(training_set[training_set.columns.difference(['label'])])

    # Load the testing sets and split into two arrays of labels and data
    testing_set = pd.read_csv('MNIST_test_HW1.csv')
    testing_labels = np.array(testing_set['label'])
    testing_data = np.array(testing_set[testing_set.columns.difference(['label'])])

    # Run the normal equation model and classify and test data
    linear_regression = LinearRegression()
    print('==================================')
    print('Normal Equation Linear Regression:')
    linear_regression.normal_equation(training_labels, training_data)
    linear_regression.classify(testing_data)
    linear_regression.verify(testing_labels)
    linear_regression.print_results()

    # Run gradient descent model and classify and test data
    linear_regression.clear()
    print('===================================')
    print('Gradient Descent Linear Regression:')
    linear_regression.gradient_descent(0.0000016, 10, training_labels, training_data)
    linear_regression.classify(testing_data)
    linear_regression.verify(testing_labels)
    linear_regression.print_results()

    # Plot out the gradient descent cost changes
    plt.plot(np.arange(len(linear_regression.gradient_descent_costs)), linear_regression.gradient_descent_costs)
    plt.show()