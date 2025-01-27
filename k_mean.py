import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

data = make_blobs(n_samples=200,centers=4,n_features=2,cluster_std=1.8,random_state=101)
print("The actual labels")
print("\n")
print(data[1])


plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
plt.show()


#Creating the clusters

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6)

kmeans.fit(data[0])
# print("The predicted labels")
# print("\n")
# print(kmeans.labels_)

#comparing the two labels

pred=kmeans.labels_

f,(ax1,ax2) = plt.subplots(2,1,sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=pred,cmap='rainbow')
ax2.set_title('Original')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
plt.show()