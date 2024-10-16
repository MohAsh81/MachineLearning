import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

# Load the digits dataset
digits = load_digits()

# Create a DataFrame
df = pd.DataFrame(digits.data)
df[64] = digits.target  # Add the target column

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    df.drop(df[64], axis=1), digits.target, test_size=0.2)

# Use DecisionTreeClassifier with max_depth=3 as the base estimator
base_estimator = DecisionTreeClassifier(max_depth=4)

# Initialize and train the AdaBoost classifier with a stronger base estimator
model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100).fit(x_train, y_train)

# Print the score of the model
print(f"The score is {model.score(x_test, y_test)}")

# Predict on the test set
y_pred = model.predict(x_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
