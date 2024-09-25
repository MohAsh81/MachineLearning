import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import *

df = pd.read_csv("income.csv")

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income']])

df['cluster'] = y_predicted


scaler = MinMaxScaler()

scaler.fit(df[['Income']])
df['Income'] = scaler.transform(df[['Income']])

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income']])
df['cluster'] = y_predicted

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
plt.scatter(df1.Age, df1['Income'], color='green')
plt.scatter(df2.Age, df2['Income'], color='red')
plt.scatter(df3.Age, df3['Income'], color='black')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[
            :, 1], color='purple', marker='*', label='centroid')
plt.legend()
plt.show()
