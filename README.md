# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset and print head,info of the dataset

2. check for null values

3. Import kmeans and fit it to the dataset

4. Plot the graph using elbow method

5. Print the predicted array

6. Plot the customer segments

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: HARSETHA J
RegisterNumber:  212223220032
*/
```

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:5])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:5])
data["Cluster"] = y_pred

plt.figure(figsize=(8, 6))
colors = ['red', 'black', 'blue', 'green', 'magenta']
for i in range(5):
    cluster = data[data["Cluster"] == i]
    plt.scatter(cluster["Annual Income (k$)"], cluster["Spending Score (1-100)"], 
                c=colors[i], label=f"Cluster {i}")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.grid(True)
plt.show()
```

## Output:
DATA.HEAD()

![444745174-cecbb6b7-0d53-409e-bae4-b93304865bb8](https://github.com/user-attachments/assets/ff5f6247-7823-4093-8cad-bc0a93226219)


DATA.INF0()

![444745220-3a258fd1-e9c2-4ef2-927e-5cf9cda73b93](https://github.com/user-attachments/assets/5838a127-a3e6-413c-b3e3-bd47f1b99ae1)


DATA.ISNULL().SUM()

![444745277-04c5f927-0271-490b-a282-fff74859d28c](https://github.com/user-attachments/assets/8e5f61b9-58b5-48fb-9217-c261b78cb3af)


PLOT USING ELBOW METHOD

![444745338-826e205f-a3b1-499c-b922-dabac7b91e7b](https://github.com/user-attachments/assets/173a6bac-75a9-4451-9c2e-530f241e737e)


CUSTOMER SEGMENT

![444745345-3d09e5bb-f87a-4d06-9388-33e77a218304](https://github.com/user-attachments/assets/22177a1e-31b5-4cd9-99c5-e1f04f9e8fad)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
