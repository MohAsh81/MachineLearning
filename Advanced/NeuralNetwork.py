import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import seaborn as sn

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

X_train = x_train / 255
X_test = x_test / 255

# Flatten the training and test data
x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)

# Define the model with one Dense layer
model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(784,), activation='relu'),  # First hidden layer
    keras.layers.Dense(10, activation='softmax')  # Output layer with softmax
])


# Compile the model with Adamax optimizer
model.compile(optimizer='Adamax',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training data
model.fit(x_train_flattened, y_train, epochs=100)

# Predict on test data
y_predicted = model.predict(x_test_flattened)

# Convert the predictions to label indices
y_predicted_labels = [np.argmax(i) for i in y_predicted]

# Compute the confusion matrix
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

# Normalize the confusion matrix
cm_normalized = cm / tf.reduce_sum(cm, axis=1)[:, np.newaxis]

# Plot the normalized confusion matrix
plt.figure(figsize=(10, 7))
sn.heatmap(cm_normalized, annot=True, fmt='.2f')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Normalized Confusion Matrix')
plt.show()
