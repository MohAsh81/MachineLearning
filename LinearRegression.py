import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('auto-mpg.data', sep="\s+", names=["mpg", "cylinders", "displacement", "horsepower",
                                                    "weight", "acceleration", "model year", "origin", "name"])

# Convert 'horsepower' to numeric, coerce errors to NaN
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Target variable (y)
y = df['mpg'].values.reshape(-1, 1)

# Feature variables (x)
x = df.drop(['mpg', 'origin', 'name'], axis=1).values

# Normalize the feature variables only
# norm_x = preprocessing.normalize(x)
scaler = preprocessing.StandardScaler()

stan_x = scaler.fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(stan_x, y, test_size=0.2, random_state=2024)

# Fit the linear regression model
model = LinearRegression().fit(x_train, y_train)

# Model score (R-squared)
print(f"Score (R-squared) is: {model.score(x_test, y_test)}")

# Predictions on the test set
y_pred = model.predict(x_test)

# Plotting actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Prediction Line')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Actual vs Predicted MPG')
plt.legend()
plt.grid(True)
plt.show()
