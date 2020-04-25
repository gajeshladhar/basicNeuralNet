import keras
import numpy as np

data=keras.datasets.boston_housing.load_data()
data=data[1]
X=data[0]
Y=data[1]

X=X-X.mean(axis=0)
u,s,vt=np.linalg.svd(X)
new_X=X.dot(vt.T[:,:3])

from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
X_reduced=pca.fit_transform(data[0])

np.sum(pca.explained_variance_ratio_)


dataset=keras.datasets.mnist.load_data()
images=dataset[1][0].reshape(10000,28*28)
labels=dataset[1][1]

pca=PCA(n_components=154)
images_reduced=pca.fit_transform(images)

from sklearn.manifold import LocallyLinearEmbedding
lle=LocallyLinearEmbedding(n_components=2,n_neighbors=10)
X_lle=lle.fit_transform(data[0])



from sklearn.manifold import TSNE
tsne=TSNE(n_components=2)
x_clusters=tsne.fit_transform(images)



import matplotlib.pyplot as plt

for i in range(0,10):
    indices=[]
    for j in range(2*5000):
        if labels[j]==i:
            indices.append(j)
    
    plt.scatter(x_clusters[(indices),0],x_clusters[(indices),1],label=str(i))
    
plt.legend()
plt.show()






