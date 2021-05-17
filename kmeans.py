import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#data from - https://www.kaggle.com/ronitf/heart-disease-uci
#load data
df = open('heart.csv')
atributes = df.readline()
data = np.loadtxt(df, delimiter = ',')
pca = PCA(2)
heart = {'data' : [], 'target' : [], 'atributes' : []}
heart['data'] = data[:,0:13]
heart['target'] = data[:,13]
heart['atributes'] = atributes.split(',')

#transform data into two dimensions
data2d = pca.fit_transform(heart['data'])

#find clusters
km = KMeans(n_clusters = 2, init = 'random', n_init = 10, max_iter = 300, tol = 1e-04, random_state = 0)
y_km = km.fit_predict(data2d)
#get cluster labels
centroids = km.cluster_centers_
print("Dane po zrzutowaniu na przestrzen wymiaru 2")
print(data2d)
print("I odpowiadajace im clustery")
print(y_km)

#visualize data
#visualize 2 clusters target == 0/1
plt.scatter(data2d[y_km == 0 , 0] , data2d[y_km == 0 , 1] , label = 'target == 0')
plt.scatter(data2d[y_km == 1 , 0] , data2d[y_km == 1 , 1] , label = 'target == 1')
#visualize centroids of those clusers
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'black', label = 'centroids')
plt.legend()
#plt.savefig('wykres', dpi = 192, format = 'png')
plt.show()
