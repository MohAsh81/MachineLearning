import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

file = pd.read_csv("titanic.csv")
file.drop(['PassengerId', 'Name', 'SibSp', 'Parch',
           'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)

target = file.Survived
inputs = file.drop("Survived", axis='columns')

le = LabelEncoder()
inputs["Sex"] = le.fit_transform(inputs['Sex'])

inputs.Age = inputs.Age.fillna(round(inputs.Age.mean(), 0))

x_train, x_test, y_train, y_test = train_test_split(
    inputs, target, test_size=0.2)

model = GaussianNB().fit(x_train, y_train)
print(model.score(x_test, y_test))
