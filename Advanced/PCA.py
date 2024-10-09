from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import pandas as pd
from matplotlib import pyplot as plt


dataset = load_digits()
dataset.keys()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

X = df
y = dataset.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=30)

model = LogisticRegression().fit(X_train, y_train)
print(model.score(X_test, y_test))


pca = PCA(0.95)
X_pca = pca.fit_transform(X)


X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=30)

model = LogisticRegression(max_iter=1000).fit(X_train_pca, y_train)
print(model.score(X_test_pca, y_test))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=30)

model = LogisticRegression(max_iter=1000).fit(X_train_pca, y_train)
print(model.score(X_test_pca, y_test))
