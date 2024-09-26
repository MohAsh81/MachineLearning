from sklearn.metrics import classification_report
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()


df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['target'] = iris.target

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

X = df.drop(['target', 'flower_name'], axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

knn = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)

print(knn.score(X_test, y_test)
      )

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


print(classification_report(y_test, y_pred))
