from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mnist = pd.read_csv('HW3/MNIST_HW3.csv')
labels = np.array(mnist.pop('label'))
features = np.array(mnist)

# Hyper Parameters
epochs = 20

input_layer_nodes = 784
hidden_layer_one_nodes = 250
output_layer_nodes = 10

# KFold Cross Validation
kfold = KFold(n_splits=5, shuffle=True)

# For each K Split fit the model and evaluate the accuracy
k_results = {
    'loss': [],
    'accuracy_avg': [],
    'accuracies': []
}

for train, test in kfold.split(features, labels):

    # Model Creation
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(input_layer_nodes, input_shape=(input_layer_nodes,))) # Input layer with no activation function
    model.add(tf.keras.layers.Dense(hidden_layer_one_nodes, activation=tf.nn.relu)) # 1 Hidden layer with a relu activation function and 250 nodes
    model.add(tf.keras.layers.Dense(output_layer_nodes, activation=tf.nn.softmax)) # Output layer with softmax to round out the final node

    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), optimizer = tf.keras.optimizers.Adam(), metrics='accuracy')

    # Fit the data to the model
    results = model.fit(features[train], labels[train], epochs=epochs, verbose=0)
    eval = model.evaluate(features[test], labels[test], verbose=0)
    
    # Save the loss and accuracy for graphical analysis
    k_results['loss'].append(eval[0])
    k_results['accuracy_avg'].append(eval[1])
    k_results['accuracies'].append(results.history['accuracy'])

averageAccuracy = np.mean(k_results['accuracy_avg'])
print(f"Average Accuracy: {(averageAccuracy*100):.2f}%")

# Display the Accuracy and Loss plots for each k-fold
fig, axs = plt.subplots(1, 1)
colors = ["red", "green", "blue", "orange", "purple"]

for i in range(5):
    axs.plot(k_results['accuracies'][i], color=colors[i])

axs.set_title('model accuracy')
axs.set_ylabel('accuracy')
axs.set_xlabel('epoch')
plt.show()