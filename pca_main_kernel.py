import numpy as np
import cdist
from scipy.spatial.distance import cdist

class KERNEL_PCA():
    def __init__(self, Xtrain, kernel_type='linear', kernel_param=1):
        ''' Apply PCA to a set of data
        Definition:  PCA(Xtrain)
        Input:       Xtrain        - array, Nxd matrix with N samples and d features
                     kernel_type   - string, either linear, gaussian or polynomial
                     kernel_param  - float, for linear none, for polynomial the degree and for gaussian gamma = 1/(2* width**2)
        '''
        self.Xtrain = Xtrain
        # Read N and d
        self.N = self.Xtrain.shape[0]
        self.d = self.Xtrain.shape[1]
        # Compute the mean
        self.mu = (1/self.N)*np.sum(self.Xtrain, axis=0)
        # Compute Covariance Matrix of original data
        self.C = (1 / (self.N - 1)) * (np.dot((self.Xtrain - self.mu).T, (self.Xtrain - self.mu)))

        #
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param


        # Compute kernel matrix
        if kernel_type == 'linear':
            K = Xtrain @ Xtrain.T

        elif kernel_type == 'polynomial':
            K = (Xtrain @ Xtrain.T + 1) ** kernel_param

        elif kernel_type == 'gaussian':
            # Use Euclidean distance, we could play with other ones
            K = cdist(Xtrain, Xtrain, 'euclidean')
            #K = np.exp(-(K ** 2) / (2. * gauss_width ** 2))
            K = np.exp(-kernel_param* K)

        else:
            print('Warning: Kernel specifications is not right')

        # Compute centered kernel matrix
        K_centered = K - np.mean(K, 0) - np.mean(K, 1)[:, np.newaxis] + np.mean(K)

        # Compute Eigen Value Decomposition of centered Kernel Matrix
        eigenValues, eigenVectors = np.linalg.eig(K_centered)

        # Sort eigenvectors & eigenvalues from biggest to smallest
        idx = eigenValues.argsort()[::-1]

        # this is  a diag matrix and we only use the real part of eigenvalues and only first d eigenvalues
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

        # Compute kernel matrix corresponding to Train data & Test Data
        if self.kernel_type == 'linear':
            K = self.Xtrain @ Xtest.T

        elif self.kernel_type == 'polynomial':
            K = (self.Xtrain @ Xtest.T + 1) ** self.kernel_param

        elif self.kernel_type == 'gaussian':
            # Use Euclidean distance, we could play with other ones
            K = cdist(self.Xtrain, Xtest, 'euclidean')
            #K = np.exp(-(K ** 2) / (2. * gauss_width ** 2))
            K = np.exp(-self.kernel_param* K)

        else:
            print('Warning: Kernel specifications is not right')

        # Compute centered kernel matrix
        K_centered = K - np.mean(K, 0) - np.mean(K, 1)[:, np.newaxis] + np.mean(K)

        # Project data onto m first principal components using the previously computed eigenvectors & eigenvalues
        # This is a mxN matrix so to get our standard mx N matrix as an output we will transpose it at the end
        X_low = self.diag_matrix[0:m, 0:m] @ self.eigenVectors[:, 0:m].T @ K_centered

        return X_low.T