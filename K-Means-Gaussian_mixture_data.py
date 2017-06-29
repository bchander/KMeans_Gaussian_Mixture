# -*- coding: utf-8 -*-
"""
K-Mens clustering on Gaussian Mixture sample data

"""


print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


np.random.seed(5)

# initializing centers of the three clusters manually
 
centers = [[1, 1], [-1, -1], [1, -1]]


''' taking the parameters for the three clusters as initialization - following 
saome gaussian distribution '''

# Try playing with mean and variance values to see how well KMeans algo clusterst the data
mu1 = [0, -2]
sig1 = [ [2, 0], [0, 3] ]

mu2 = [5, 0]
sig2 = [ [3, 0], [0, 1] ]

mu3 = [3, 0]
sig3 = [ [1, 0], [0, 4] ]

''' sampling points (x,y) from three diff gaussian distributions '''

X1, y1 = np.random.multivariate_normal(mu1, sig1, 100).T
X2, y2 = np.random.multivariate_normal(mu2, sig2, 100).T
X3, y3 = np.random.multivariate_normal(mu3, sig3, 100).T    

''' clubbing all the data sampled from the three diff distributions'''
                           
x = np.concatenate((X1, X2, X3))
y = np.concatenate((y1, y2, y3))

x= x.reshape(300,1)
y= y.reshape(300,1)

X = np.column_stack((x,y))

labels = ([1] * 100) + ([2] * 100) + ([3] * 100)

# Plot the data to visualize 

fig = plt.figure()
plt.scatter(x, y, 17, c=labels) # color criteria is based on the labels of the data
#fig.savefig("Actual_data_from_gaussian_mixture.png")


''' Inititalizing KMeans estimator with three clusters'''

estimator = KMeans(n_clusters=3,max_iter =100)

''' fitting and pedicting lables using the data points we have. Note that prediction 
from the KMeans algo. will be the labels assigned to them'''

esti_fit = estimator.fit(X)
y_predict = esti_fit.predict(X)

#print y_predict

'''plotting the clustered data and saving that in a file'''

fig = plt.figure()
plt.scatter(x,y, 17, c=y_predict)
plt.title('Kmean for cluster size = 3')
#fig.savefig("After_KMeans_cluster_size_3.png")



'''

Intro - 

k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster.
This results in a partitioning of the data space into Voronoi cells.

The problem is computationally difficult (NP-hard); however, there are efficient heuristic algorithms that are commonly employed and converge quickly to a local optimum. 

These are usually similar to the expectation-maximization algorithm for mixtures of Gaussian distributions via an iterative refinement approach employed by both algorithms. 
Additionally, they both use cluster centers to model the data; however, k-means clustering tends to find clusters of comparable spatial extent, while the expectation-maximization mechanism allows clusters to have different shapes.
One can apply the 1-nearest neighbor classifier on the cluster centers obtained by k-means to classify new data into the existing clusters. This is known as nearest centroid classifier or Rocchio algorithm.

Commonly used initialization methods are Forgy and Random Partition.[7] 
The Forgy method randomly chooses k observations from the data set and uses these as the initial means. 
The Random Partition method first randomly assigns a cluster to each observation and then proceeds to the update step, thus computing the initial mean to be the centroid of the cluster's randomly assigned points. 
The Forgy method tends to spread the initial means out, while Random Partition places all of them close to the center of the data set.

in computer graphics, color quantization is the task of reducing the color palette of an image to a fixed number of colors k. The k-means algorithm can easily be used for this task and produces competitive results. 

That set of points (Centroids of K) are called seeds, sites, or generators just like in voronoi diag

'''


