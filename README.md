
Kernel PCA implementation from scratch in Python using NumPy
# PCA
Principal Component Analysis is one of the most classis techniques in Machine Learning. It dates back to 1901 The implementation of PCA seems quite straightforward.
It mainly follows the next steps:

- We have a data matrix **X** with *n* datapoints (rows) and *d* features per datapoint (columns). We need to select number of components *m* we wish to mantain.

1. At first, we compute the mean vector **μ** which has *d* components. In other words, we take the mean over each columns of our data matrix **X**
2. We center the data by substracting the mean from each data point of **X**. We have, **X<sub>c</sub>** = **X** - **μ**
3. Compute Covariance matrix: **C** =(1/(*n*-1)) **X<sub>c</sub>'X<sub>c</sub>**  where (**X<sub>c</sub>'** is transpose of **X<sub>c</sub>**)
   The Covariance Matrix has dimension *d*x*d*
5. Compute Eigenvalue Decomposition of **C** = **V** **Σ** **V'** and make sure that eigenvectors & values pairs are ordered in an ascending manner.
6. Mantain only the first *m* eigenvectors of **V** yielding **V<sub>p</sub>** = **V[:, 0:m]**
7. Project some new data **X<sub>new</sub>** to low dimensional space using our first *m* eigenvectors. Before doing this, we need to substract the mean **μ** we previously computed in (1) from **X<sub>new</sub>**. Therefore: **X<sub>new, c</sub>** = **X<sub>new</sub>** - **μ**  and finally: **X<sub>low</sub>** = **X<sub>new, c</sub>**  **V[:, 0:m]**

PCA is a linear technique and therefore works best when we have linear data. However, when facing non linear data we may not be successful. One of the most elegant tricks in ML is the famous *kernel trick* . 

- The *kernel trick* basically replaces **XX'** with a kernel, also known as similarity matrix, **K**. The big advantage when kernelizing an algorithm is that we do not need to compute high dimensional feature vectors Φ(x), but can still take advantage of them. k(x<sub>i</sub>, x<sub>j</sub>) = Φ(x<sub>i</sub>)' Φ(x<sub>j</sub>) but we have a formula for  k(x<sub>i</sub>, x<sub>j</sub>) and do not need to compute any Φ(x).
- The Gaussian kernel is one of the most popular ones. One can show that it corresponds to an infinite dimensional feature space. In other words, *kernel trick* makes our algorithms much more flexible and powerful.
-  In order to kernelize an algorithm, we need to formulate the algorithm such that our data matrix **X** only appears as **XX'** throughout all the algorithm.

In other classical ML techniques as linear regression or support vector machines (SVMs), kernelizing the algorithm is pretty clear since the formulations directly involve just  **XX'**. In the case of linearr regrerssion this leads to kernelized linear regression. 

# Kernel PCA

We may remember that in PCA we worked with the eigendecomposition of the covariance matrix **C =(1/(*n*-1)) X'X** . Be careful, this is not the term we need ( **XX'**) to kernelize our algortihm but it looks very similar :) . It is not quite obvious how we should kernelize PCA. The good new is: some clever people worked on this 20 years ago and figured out an elegant solution. 
