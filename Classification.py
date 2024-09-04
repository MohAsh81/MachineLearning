import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *

# Read CSV file
file = pd.read_csv('wdbc.data', sep=',')

# Select first 11 columns
file = file.iloc[:, :11]

# Assign headers to the DataFrame
file.columns = [
    'Id', 'Diagnosis', 'radius', 'texture', 'perimeter', 'area', 'smoothness',
    'compactness', 'concavity', 'concave_points', 'symmetry'
]

# Separate datas
y = file['Diagnosis']
x = file.drop(['Diagnosis', 'Id'], axis=1)

# Scaling data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y,test_size=0.1)

# Train the model
model = LogisticRegression().fit(x_train,y_train)
y_pred = model.predict(x_test)

# Evaluation of model
r_squared = model.score(x_test,y_test)
print(f"R^2 is {r_squared}")