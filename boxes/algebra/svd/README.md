# Algebra : SVD

Singular-value decomposition is useful, but algorithmically opaque. It is often just presented as a bunch of linear algebra manipulations. Here we show the algorithm the underlies the computation of SVD and make some general claims about eigen decomposition.

- Eigenvector and Eigenvalues of Covariance Matrix is PCA

## Power Iteration Algorithm

Repeatedly apply the matrix to a random vector, normalize and continue. This will converge to the largest eigenvector. Subtract this to compute subsequent (smaller) eigen vectors.

