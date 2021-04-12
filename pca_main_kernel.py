import numpy as np
from sklearn.datasets import make_circles
import cdist
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_moons

class KERNEL_PCA():
    def __init__(self, Xtrain, kernel_type='linear', pol_degree=2, gauss_width=1e-05):
        ''' Apply PCA to a set of data
        Definition:  PCA(Xtrain)
        Input:       Xtrain        - array, Nxd matrix with N samples and d features
        '''
        self.Xtrain = Xtrain
        # Read N and d
        self.N = self.Xtrain.shape[0]
        self.d = self.Xtrain.shape[1]
        # Compute the mean
        self.mu = (1/self.N)*np.sum(self.Xtrain, axis=0)
        # Compute Covariance Matrix of original data
        self.C = (1 / (self.N - 1)) * (np.dot((self.Xtrain - self.mu).T, (self.Xtrain - self.mu)))

        # Compute kernel matrix
        if kernel_type == 'linear':
            K = Xtrain @ Xtrain.T

        elif kernel_type == 'polynomial':
            K = (Xtrain @ Xtrain.T + 1) ** pol_degree

        elif kernel_type == 'gaussian':
            # Use Euclidean distance, we could play with other ones
            K = cdist(Xtrain, Xtrain, 'euclidean')
            #K = np.exp(-(K ** 2) / (2. * gauss_width ** 2))
            K = np.exp(-gauss_width* K)

        else:
            print('Warning: Kernel specifications is not right')

        # Compute centered kernel matrix
        K_centered = K - np.mean(K, 0) - np.mean(K, 1)[:, np.newaxis] + np.mean(K)

        # Compute Eigen Value Decomposition of centered Kernel Matrix
        eigenValues, eigenVectors = np.linalg.eig(K_centered)

        # Sort eigenvectors & eigenvalues from biggest to smallest
        idx = eigenValues.argsort()[::-1]

        # this is  a diag matrix and only real part of eigenvalues and only first d eigenvalues
        eigenValues = np.real(eigenValues[idx])[0:self.d]

        self.diag_matrix = np.diag(1./(np.sqrt(eigenValues)))
        self.eigenVectors = np.real(eigenVectors[:, idx])

        self.K_train = K_centered



    def project(self, Xtest, m):
        ''' Project data from Xtest into a lower m dimensional space with the correspondent eigenvectors
        Definition:  z = project(self, Xtest, m)
        Input:       Xtest    - DxN array of N data points with D features
                     m        - int, dimension of the subspace, number of principal components used to project
        Output:      z        - mxN array of N data points with reduced dimensionality m
        '''

        Xtest = 2
        X_low = self.diag_matrix[0:m, 0:m] @ self.eigenVectors[:, 0:m].T @ self.K_train

        return X_low.T

        # Compute Eigenvalues and eigenvectors of C
#        self.D, self.U = la.eig(self.C)
#        # Real part of the Eigenvalues
#        self.D = np.real(self.D)

# Load dataset
X, y = make_moons(n_samples=100, random_state=123)

# Apply kernel PCA
kpca = KERNEL_PCA(Xtrain=X, kernel_type='polynomial', gauss_width=2, pol_degree=5)
X_pc = kpca.project(X, 2)

plt.figure(figsize=(8,6))
plt.scatter(X_pc[y==0, 0], X_pc[y==0, 1], color='red', alpha=0.5)
plt.scatter(X_pc[y==1, 0], X_pc[y==1, 1], color='blue', alpha=0.5)

plt.title('First 2 principal components after RBF Kernel PCA')
plt.text(-0.18, 0.18, 'gamma = 0.1', fontsize=12)
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.figure(figsize=(8,6))
plt.scatter(X_pc[y==0, 0], np.zeros((50)), color='red', alpha=0.5)
plt.scatter(X_pc[y==1, 0], np.zeros((50)), color='blue', alpha=0.5)

plt.title('First principal component after RBF Kernel PCA')
plt.text(-0.17, 0.007, 'gamma = 15', fontsize=12)
plt.xlabel('PC1')
plt.show()

'''
X = np.random.randn(20, 3)
X, y = make_circles(n_samples=400, factor=.3, noise=.05)
example =  KERNEL_PCA(Xtrain=X, kernel_type='gaussian', gauss_width=0.4)
X_kpca = example.project(Xtest=X, m=2).T

print(X.shape)

plt.figure()
plt.subplot(2, 2, 1, aspect='equal')
plt.title("Original space")
reds = y == 0
blues = y == 1

plt.scatter(X[reds, 0], X[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.subplot(2, 2, 3, aspect='equal')
plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Projection by KPCA")
plt.xlabel(r"1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")

plt.show()

'''


