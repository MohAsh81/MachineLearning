from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn import tree
from sklearn.model_selection import train_test_split

# Reading Titanic dataset
df = pd.read_csv("titanic.csv")

# Dropping unnecessary columns
df = df.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"], axis=1)

# Filling missing values in Age with the mean
df["Age"] = df["Age"].fillna(round(df["Age"].mean(), 2))

# Encoding 'Sex' column
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])

# Separating features and target variable
x = df.drop(["Survived"], axis=1)
y = df["Survived"]

# Normalizing the feature set
norm_df = normalize(x)
x_norm = pd.DataFrame(norm_df, columns=x.columns)

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.2, random_state=2024)

# Training the decision tree model
model = tree.DecisionTreeClassifier().fit(x_train, y_train)

# Printing the model score
print(f"The score is {model.score(x_test, y_test)}")

# Plotting feature importance
importance = model.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(x.columns, importance, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance of Decision Tree Model')
plt.show()
