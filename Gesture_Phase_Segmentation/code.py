# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 20:32:45 2018

@author: utsav
"""
import time
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

from sklearn.neighbors import KNeighborsClassifier

######Importing Data from excel sheet
df = pd.read_excel("data.xlsx", sheet_name='Sheet1',dtype=np.float64)
data = df.values

df = pd.read_excel("data.xlsx", sheet_name='Sheet2')
data_class = df.values
data_class = data_class[:,1]

#############Instances of each class
class_count = []
class_count.append((data_class == 'D').sum())
class_count.append((data_class == 'R').sum())
class_count.append((data_class == 'P').sum())
class_count.append((data_class == 'S').sum())
class_count.append((data_class == 'H').sum())

#####Applying dimensionality reduction
pca = PCA(n_components=3).fit(data)
reduced_data = pca.transform(data)

################Meanshift
bandwidth = estimate_bandwidth(reduced_data, quantile=0.15)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(reduced_data)
labels_ms = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels_ms)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

########Plot MeanShift
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels_ms == k
    cluster_center = cluster_centers[k]
    plt.plot(reduced_data[my_members, 0], reduced_data[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=8)
plt.title("Clustering using Mean Shift")
plt.axis("off")
plt.show()
silhouette_ms = 0
silhouette_ms = metrics.silhouette_score(reduced_data, labels_ms, metric='euclidean')
calinski_ms = 0
calinski_ms = metrics.calinski_harabaz_score(reduced_data, labels_ms)

labels_ms_count = []
labels_ms_count.append((labels_ms == 0).sum())
labels_ms_count.append((labels_ms == 1).sum())
labels_ms_count.append((labels_ms == 2).sum())
labels_ms_count.append((labels_ms == 3).sum())
labels_ms_count.append((labels_ms == 4).sum())

######Different number of clusters and analysis
silhouette_kmeans = []
calinski_kmeans = []
time_kmeans = []

silhouette_h = []
calinski_h = []
time_h = []

end_1 = time.time()
for i in range(3,8):
	
	#######K-means
	start_1 = time.time()
	kmeans = KMeans(n_clusters=i).fit(reduced_data)
	y_pred = kmeans.fit_predict(reduced_data)

	
	#######Plot the clusters
	centroids = kmeans.cluster_centers_
	pcac = PCA(n_components=3).fit(centroids)
	new_c = pcac.transform(centroids)
	plt.plot(new_c[:,0], new_c[:,1],c='black',marker='x',markersize=20,linestyle="None")
	plt.scatter(reduced_data[:,0], reduced_data[:,1],c=y_pred,marker='_')
	plt.title("Clustering using K-means")
	plt.axis("off")
	plt.show()
	
	########3DPlot
	fig = plt.figure(figsize=(4, 3))
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	ax.scatter(reduced_data[:, 1], reduced_data[:, 0], reduced_data[:, 2],c=y_pred)
	ax.plot(new_c[:,1], new_c[:,0],new_c[:,2],c='black',marker='x',linestyle="None")
	plt.title("Clustering using K-means")

	plt.show()
		
	#########Indices
	labels_kmeans = kmeans.labels_
	
	silhouette_kmeans.append(metrics.silhouette_score(reduced_data, labels_kmeans, metric='euclidean'))
	calinski_kmeans.append(metrics.calinski_harabaz_score(reduced_data, labels_kmeans))
	end_1 = time.time()
	time_kmeans.append(end_1 - start_1)
	
	################Hierarchical clustering
	start_1 = time.time()
	clustering = AgglomerativeClustering(n_clusters=i).fit(reduced_data)
	h_pred = clustering.fit_predict(reduced_data)
	
	###############Plot
	fig = plt.figure(figsize=(4, 3))
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	ax.scatter(reduced_data[:, 1], reduced_data[:, 0], reduced_data[:, 2],c=h_pred)
	plt.title("Clustering using Hierarchical clustering")
	plt.show()
		
	#########Indices
	labels_h = clustering.labels_
	silhouette_h.append(metrics.silhouette_score(reduced_data, labels_h, metric='euclidean'))
	calinski_h.append(metrics.calinski_harabaz_score(reduced_data, labels_h))
	end_1 = time.time()
	time_h.append(end_1 - start_1)

x_axis = [3,4,5,6,7]
plt.title("Time using K-means clustering")
plt.plot(x_axis,time_kmeans)
plt.scatter(x_axis,time_kmeans)
plt.legend()
plt.xlabel("Number of clusters")
plt.ylabel("Time")
plt.show() 

plt.title("Time using Hierarchical clustering")
plt.plot(x_axis,time_h)
plt.scatter(x_axis,time_h)
plt.legend()
plt.xlabel("Number of clusters")
plt.ylabel("Time")
plt.show() 

##########Classification using KNN
train_without_KNN = KNeighborsClassifier(n_neighbors=3)
train_without_KNN.fit(reduced_data[0:1980],data_class[0:1980]) 
x = train_without_KNN.score(reduced_data[1980:],data_class[1980:]) *100
print("Accuracy", x)