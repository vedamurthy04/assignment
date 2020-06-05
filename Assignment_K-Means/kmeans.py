import os
import struct
import numpy as np
import pandas as pd
import sklearn.datasets
import ipyvolume as ipv
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from ipywidgets import ColorPicker, VBox, \
    interact, interactive, fixed



def compute_bic(kmeans,x):

    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    m = kmeans.n_clusters
    n = np.bincount(labels)
    N, d = x.shape
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(x[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])
    const_term = 0.5 * m * np.log(N) * (d+1)
    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)



url = 'https://raw.githubusercontent.com/MSPawanRanjith/FileTransfer/master/kmean_dataset.csv'
df = pd.read_csv(url, error_bad_lines=False)
df.head(10)


x = df.iloc[:, [0,1,2]].values
kmeans5 = KMeans(n_clusters=5)
y_kmeans5 = kmeans5.fit_predict(x)
print(y_kmeans5)

kmeans5.cluster_centers_




Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()




print("the rapid changes are made between 2 and 4. so we take the average value for k as 3 and proceed")


kmeans3 = KMeans(n_clusters=3)
y_kmeans3 = kmeans3.fit_predict(x)
print(y_kmeans3)

kmeans3.cluster_centers_
plt.scatter(x[:,0],x[:,1], c=y_kmeans3,cmap='rainbow')
plt.show()


print("BIC FUNCTION TO FIND THE VALUES OF BIC")

ks = range(1,9)
KMeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(x) for i in ks]
BIC = [compute_bic(kmeansi,x) for kmeansi in KMeans]
print(BIC)
print (max(BIC))
print("the optimum value for k using BIC is",BIC.index(max(BIC))+1)

plt.plot(ks,BIC,'r-o')
plt.title(" data  (k vs BIC)")
plt.xlabel("# k values")
plt.ylabel("# BIC")
plt.show()