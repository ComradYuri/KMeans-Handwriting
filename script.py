import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
print(digits.data)
print(digits.target)
print(np.unique(digits.target))

# print digit at index 10
plt.gray()
plt.matshow(digits.images[10])
plt.show()
plt.close('all')

model = KMeans(n_clusters=len(np.unique(digits.target)))
model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()
