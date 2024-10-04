from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv("iris.data", sep=',', header=None, names=[
                 "sepal length", "sepal width", "petal length", "petal width", "target"])

le = LabelEncoder()

df["target"] = le.fit_transform(df["target"])

x = df.drop(["target"], axis=1)
y = df.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2022)

model = SVC().fit(x_train,y_train)

print(f"The score is: {model.score(x_test,y_test)}")