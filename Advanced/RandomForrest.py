import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

digits = load_digits()

df = pd.DataFrame(digits.data)
df[64] = digits.target

x_train, x_test, y_train, y_test = train_test_split(
    df.drop(df[64], axis=1), digits.target, test_size=0.2)

model = RandomForestClassifier(n_estimators=30).fit(x_train,y_train)
print(f"The score is {model.score(x_test,y_test)}")

y_pred = model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()