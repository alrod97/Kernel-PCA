from pca_main_kernel import KERNEL_PCA
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

# Here we can apply our PCA Kernel implementation to some example datasets from Sklearn

# Load dataset moon from Sklearn
X, y = make_moons(n_samples=200, random_state=123)


# Perform standard PCA to this data set. In this case, we just need to set a linear kernel projecting first 2 principal
# components
pca_standard= KERNEL_PCA(Xtrain=X, kernel_type='linear', kernel_param=None)
X_pca = pca_standard.project(Xtest=X, m=2)

# Now we just keep the first principal component
X_pca_onedim = pca_standard.project(Xtest=X, m=1)


# Perform kernel PCA to this data set. In this case, we set a Gaussian (RBF) kernel projecting first 2 principal
# components with kernel parameter gamma = 15
kernel_pca = KERNEL_PCA(Xtrain=X, kernel_type='gaussian', kernel_param=15)
X_kpca_twodim = kernel_pca.project(Xtest=X, m=2)

# Now we just keep the first principal component as before
X_kpca_onedim = kernel_pca.project(Xtest=X, m=1)

# Plots

plt.figure(figsize=(8,6))

plt.scatter(X[y==0, 0], X[y==0, 1], color='red', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', alpha=0.5)

plt.title('Moon Ddataset')
plt.ylabel('y coordinate')
plt.xlabel('x coordinate')

plt.figure(figsize=(8,6))
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='red', alpha=0.5)
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='blue', alpha=0.5)

plt.title('First 2 principal components after Linear/Standard PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')


plt.figure(figsize=(8,6))
plt.scatter(X_pca_onedim[y==0, 0], np.zeros((100,1)), color='red', alpha=0.5)
plt.scatter(X_pca_onedim[y==1, 0], np.zeros((100,1)), color='blue', alpha=0.5)

plt.title('First  principal component after Linear/Standard PCA')
plt.xlabel('PC1')
plt.ylabel('0')

plt.figure(figsize=(8,6))
plt.scatter(X_kpca_twodim[y==0, 0], X_kpca_twodim[y==0, 1], color='red', alpha=0.5)
plt.scatter(X_kpca_twodim[y==1, 0], X_kpca_twodim[y==1, 1], color='blue', alpha=0.5)

plt.title('First 2 principal components after Kernel PCA with Gaussian kernel & Gamma=15')
plt.xlabel('PC1')
plt.ylabel('PC2')


plt.figure(figsize=(8,6))
plt.scatter(X_kpca_onedim[y==0, 0], np.zeros((100,1)), color='red', alpha=0.5)
plt.scatter(X_kpca_onedim[y==1, 0], np.zeros((100,1)), color='blue', alpha=0.5)

plt.title('First  principal component after Kernel PCA with Gaussian kernel & Gamma=15')
plt.xlabel('PC1')
plt.ylabel('0')
plt.show()