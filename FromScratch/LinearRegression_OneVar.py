import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('Day.csv')


def cost_function(m, b, data):
    total_error = 0
    for i in range(len(data)):
        x = data.iloc[i].temp
        y = data.iloc[i].cnt
        total_error += (y - ((m * x) + b)) ** 2
    return total_error / len(data)


# Function for gradient descent
def gradient_descent(m_now, b_now, points, lr):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    # Calculating gradients
    for i in range(n):
        x = points.iloc[i].temp
        y = points.iloc[i].cnt
        m_gradient += -(2 / n) * x * (y - ((m_now * x) + b_now))
        b_gradient += -(2 / n) * (y - ((m_now * x) + b_now))

    # Updating m and b
    m = m_now - lr * m_gradient
    b = b_now - lr * b_gradient

    return m, b


# Initial values
m = 0
b = 0
lr = 0.07
tolerance = 1e-6  # Set a small tolerance to stop when the error stabilizes
epochs = 0
error_list = []

# Initialize first error
prev_error = cost_function(m, b, data)

# Running gradient descent
while True:
    m, b = gradient_descent(m, b, data, lr)
    current_error = cost_function(m, b, data)

    # Break the loop if improvement is smaller than tolerance
    if abs(prev_error - current_error) < tolerance:
        break

    # Store the error and update the previous error
    error_list.append(current_error)
    prev_error = current_error
    epochs += 1
    print(f"Epoch: {epochs}, m: {m:.4f}, b: {b:.4f}, error: {current_error:.6f}")

# Generate data for the regression line
x_vals = np.linspace(data.temp.min(), data.temp.max(), 100)
y_vals = m * x_vals + b

# Plot the results
plt.figure(figsize=(12, 5))

# Subplot 1: Regression line with scatter plot
plt.subplot(1, 2, 1)
plt.scatter(data.temp, data.cnt, color='green', label="Data Points")
plt.plot(x_vals, y_vals, color='red', label="Regression Line")
plt.xlabel("Temp")
plt.ylabel("Cnt")
plt.legend()

# Subplot 2: Error reduction over epochs
plt.subplot(1, 2, 2)
plt.scatter(range(len(error_list)), error_list, s=10, color='blue')
plt.xlabel("Epoch")
plt.ylabel("Error")

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()


# 2272020.255832