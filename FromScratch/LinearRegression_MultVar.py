import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the data from the CSV file (assuming it contains multiple features)
data = pd.read_csv('Day.csv')

# Prepare the features (X) and the target (y)
# Select multiple features, e.g., 'temp', 'humidity', 'windspeed'
X = data[['temp', 'hum', 'windspeed']].values
y = data['cnt'].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add a column of ones to X to account for the bias term (b)
# Now X has a column for the bias (intercept)
X = np.c_[np.ones(X.shape[0]), X]

# Initialize parameters (weights for each feature + bias)
w = np.zeros(X.shape[1])  # One weight for each feature + bias term
lr = 0.01  # Learning rate (adjusted to be smaller)
tolerance = 1e-6  # Tolerance to stop when improvement is small
epochs = 0  # Count the number of epochs
error_list = []  # Track errors over time


# Define the cost function (mean squared error)
def cost_function(X, y, w):
    n = len(y)
    predictions = X.dot(w)
    error = predictions - y
    cost = (1 / (2 * n)) * np.dot(error.T, error)
    return cost


# Define the gradient descent algorithm
def gradient_descent(X, y, w, lr):
    n = len(y)
    predictions = X.dot(w)
    error = predictions - y
    gradient = (1 / n) * X.T.dot(error)
    w = w - lr * gradient
    return w


# Initialize the first error
prev_error = cost_function(X, y, w)

# Gradient descent loop
while True:
    w = gradient_descent(X, y, w, lr)
    current_error = cost_function(X, y, w)

    # Break if improvement is smaller than the tolerance
    if abs(prev_error - current_error) < tolerance:
        break

    # Store the error and update the previous error
    error_list.append(current_error)
    prev_error = current_error
    epochs += 1
    print(f"Epoch: {epochs}, Weights: {w}, Error: {current_error:.6f}")

# Generate predictions for plotting (only plotting one feature against the target)
x_vals = np.linspace(data.temp.min(), data.temp.max(), 100)

# Since we standardized the data, we need to transform the plotting data
# Assume other features fixed for plotting
x_vals_scaled = scaler.transform(np.c_[x_vals, np.zeros((100, 2))])
y_vals = w[1] * x_vals_scaled[:, 0] + w[0]  # Temp coefficient + bias term

# Plot the results
plt.figure(figsize=(12, 5))

# Subplot 1: Regression line with scatter plot (for 'temp')
plt.subplot(1, 2, 1)
plt.scatter(data.temp, data.cnt, color='green', label="Data Points")
plt.plot(x_vals, y_vals, color='red', label="Regression Line (Temp)")
plt.xlabel("Temp")
plt.ylabel("Cnt")
plt.legend()

# Subplot 2: Error reduction over epochs
plt.subplot(1, 2, 2)
plt.plot(range(len(error_list)), error_list, color='blue')
plt.xlabel("Epoch")
plt.ylabel("Error")

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
