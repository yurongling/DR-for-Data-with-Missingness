import numpy as np
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression


def PCA_reduce(X, Q):
    """
    A helpful function for linearly reducing the dimensionality of the data X
    to Q.
    :param X: data array of size N (number of points) x D (dimensions)
    :param Q: Number of latent dimensions, Q < D
    :return: PCA projection array of size N x Q.
    """
    assert Q <= X.shape[1], 'Cannot have more latent dimensions than observed'
    evecs, evals = np.linalg.eigh(np.cov(X.T))
    i = np.argsort(evecs)[::-1]
    W = evals[:, i]
    W = W[:, :Q]
    return (X - X.mean(0)).dot(W)

def BC_mdsReduce(Y,M,k='CNG',prop=0.95): 
    """
    A function returns low-dimensional data array of size N by k.
    
    Args:
    Y: data array of size n samples by n features.

    M: N-by-D indicator array with 1 indicating non-missingness and 0 missingness. 

    k: int or str, indicating the dimension k to be extracted; k='all', the dimensions with the positive eigenvalues would be retained;
    k='CNG', k is determined by the CNG scree test;
    k='proportion', k is determined by the proportion of variance explained (prop). Default='CNG'.
    k='all', k is number of eigenvalues that are non-negative.

    prop: float in (0,1], default=0.95. The proportion of variance that extracted low-dimensional data possess.
    """
    if type(k) == int :
        assert k <= Y.shape[0], 'Cannot have more latent dimensions than observed'
    if k == 'proportion':
        assert prop>1 or prop <=0, 'Proportion of variance should lie in the range (0,1]'

    N,D = Y.shape
    G = BC_Gram(Y, M) # Get the bias-corrected Gram matrix.
       
    evalu,evect = np.linalg.eigh(G)
    i = np.argsort(evalu)[::-1]
    E = evalu[i]

    idx_positive = (E>=0)
    sum_evalu = E[idx_positive].sum()
    
    if k=='all':
        k = idx_positive.sum()
    
    if k == 'CNG': # k is determined by the CNG scree test.
        k = nCng(E)
        
    elif k == 'proportion': # k is determined by the proportion of variance explained.
        tmp = E[0]
        idx = 0
        while tmp/sum_evalu <= prop:
            idx = idx+1
            tmp += E[idx]
        k = idx + 1
    print('[BC-PCA]: the first '+str(k)+' components were extracted with the proportion of variance explained: ' + \
              str(np.sum(E[:k])/sum_evalu))
    
    E = np.diag(E[:k])
    W = evect[:,i]
    W = W[:,:k]
    return W.dot(np.sqrt(E))

def Euc_to_Gram(E):
    """
    Convert Squared Euclidean distance matrix to Gram matrix.
    
    """
    
    n = E.shape[0]# E is the squared dissimilarity matrix
    G = np.identity(n) - 1/n * np.dot(np.ones((n,1)), np.ones((1,n)))
    G = -0.5*np.dot(np.dot(G, E), G)
    
    return G

def mdsReduce(Y,k='CNG',prop=0.95,dissimilarity=None):
    """
    Classical multidimensional scaling equivalent to PCA.

    Args:
    Y: data array of size n samples by n features or n samples by n samples.
    
    k: int or str, indicating the dimension to be extracted; k='all', the dimensions with the positive eigenvalues would be retained;
    k='CNG', the dimension k is determined by the CNG scree test;
    k='proportion', the dimension k is determined by the proportion of variance explained (prop). Default='CNG'.
    
    prop: float in (0,1], default=0.95. The proportion of variance that extracted low-dimensional data possess.

    dissimilarity: str, default=None. For dissimilarity='precomputed', Y should be a squared dissimilarity matrix (n samples by n samples).

    """
    if type(k) == int :
        assert k <= Y.shape[0], 'Cannot have more latent dimensions than observed'
    if k == 'proportion':
        assert prop>1 or prop <=0, 'Proportion of variance should lie in the range (0,1]'
    if dissimilarity == 'precomputed': # Convert a squared dissimilarity matrix to a Gram matrix
        G = Euc_to_Gram(Y)
    else:
        Yn =  Y - np.mean(Y, axis=0)
        G = Yn.dot(Yn.T)

    evalu,evect = np.linalg.eigh(G)
    i = np.argsort(evalu)[::-1]
    E = evalu[i]
    sum_evalu = E.sum()
    
    if k=='all':
        k = np.sum(evalu >= 0 )
        
    if k == 'CNG': # k is determined by the CNG scree test.
        k = nCng(E)
        
    elif k == 'proportion': # k is determined by the proportion of variance explained.
        tmp = E[0]
        idx = 0
        while tmp/sum_evalu <= prop:
            idx = idx+1
            tmp += E[idx]
        k = idx + 1
    print('[PCA]: the first '+str(k)+' components were extracted with the proportion of variance explained: ' + \
              str(np.sum(E[:k])/sum_evalu))
    E = np.diag(E[:k])
    W = evect[:,i]
    W = W[:,:k]
    
    return W.dot(np.sqrt(E))
        
def nCng(x, order = 'descending'):
    """
    The slope of all possible sets of three adjacent eigenvalues are compared,
    so \emph{CNG} indices can be applied only when more than six eigenvalues are
    used. The eigenvalue at which the greatest difference between two successive
    slopes occurs is the indicator of the number of components/factors to
    retain.
    
    x: array of eigenvalues in descending/ascending order, otherwise should be sorted.
    
    return:
    number of components/factors to retain
    """
    
    n_length = 2
    n = np.sum(x>=0) # exclude the negative eigenvalues

    #if n<x.shape[0]:
    #    print('there exist negative eigenvalues.')
    
    assert n >= 6, 'The number of variables must be at least 6.'
    
    if order =='ascending':
        x = x[::-1]
        
    elif order == 'nonsorted':
        x = np.sort(x)[::-1]
    
    i = 0
    cng = np.zeros((n-5,))
    while (i+2*n_length+1) <= (n-1):
        xa = np.arange(i, i+n_length+1).reshape(-1,1)
        ya = x[xa]
        compa = LinearRegression().fit(xa, ya).coef_
        
        xb = np.arange(i+n_length+1, i+2*n_length+2).reshape(-1,1)
        yb = x[xb]
        compb = LinearRegression().fit(xb, yb).coef_
        
        cng[i] = compb - compa
        i +=1
        
    cng = np.argmax(cng)+n_length
    
    return cng


def BC_Gram(Y,M):
    """
    Return a bias-corrected Gram matrix with the missing positions known.
    
    Input:
    Y- array, shape (n_samples, n_features).
    M- N by D indicator array with 1 indicating non-missingness and 0 missingness. 
    
    """    
    N,D = Y.shape

    pd = (M != 0).mean(axis=0)
    Yn =  Y - np.mean(Y, axis=0)/pd 
    Yn = M*Yn
    G = Yn.dot(Yn.T) 
    
    c = (M != 0).mean() 
    pd = (M != 0).mean(axis=0)
    pr = (M != 0).mean(axis=1)
    mod = np.outer(pr,pd)
    
    if np.max(mod) >= c:
        c = np.max(mod)
        #print('not moment matching!!')
    #print(c)      
    mod = mod/c
    mod = mod.dot(mod.T)

    for i in range(N):
        mod[i,i] = np.sum(pr[i]*pd)/c
  
    G = G/mod    

    return G

def estimate_gamma(X, k=7):
    """
    Estimate the gamma parameter in the spectral clustering where
    delta = average Euclidean distance of samples to their k-th nearest neighbor,
    where affinity matrix  = np.exp(- dist_matrix ** 2 / (2. * delta ** 2))
    gamma = 1 / (2. * delta ** 2)
    """ 
    D = euclidean_distances(X, X)
    sort_D = np.sort(D, axis=1)
    delta = np.mean(sort_D[:, k-1])
    gamma = 1 / (2. * delta **2)
    
    return gamma


def id_consensus(Y, n_cluster, threshold=0.9, method=["KMeans"]):
    """ A function for differentiating dropouts(missingness) from true biological expression.
    Args:
    Y - array, shape (n_samples, n_features).
    n_cluster - integer or array of integers, which indicates the number of clusters. 
    threshold - float between 0 and 1. If most expression values (>=threshold) for a gene in 
    the correspionding cluster are 0, then we conclude that zeros for the gene in the cluster 
    are true non-expression. Otherwise they are dropouts.
    method - list of strings, indicating the name of the clustering method in sklearn. 
    
    Return an indicator array of size n_samples by n_features with integer 0 indicating dropouts (missingness) 
    and integer 1 representing true biological expression.
    """
    
    assert threshold >=0 and threshold <=1, "Threshold cannot be less than 0 or greater than 1."
    if type(n_cluster) == int:
        n_cluster = np.array([n_cluster]).astype(int)

    k_neighbor = 7 # keighbor size for determining the gamma parameter in the spectral clustering.
    ite_num = 30 # Number of time the k-means algorithm will be run with different centroid seeds

    K1 = n_cluster.shape[0]
    K2 = len(method)
    N,D = Y.shape
    candi_matrix = np.zeros((N, D, K1, K2), dtype=int)
    idx_matrix = np.zeros((N, D),dtype=int)
    idx_non_zero = (Y!=0)
    
    if 'SpectralClustering' in method:
        gamma = estimate_gamma(Y, k_neighbor)
    
    for tt in range(K2):
        for i in range(K1):
            if method[tt] == 'SpectralClustering':
                clustering = getattr(cluster, method[tt])(n_clusters = n_cluster[i],  \
                                              gamma = gamma,n_init = ite_num).fit(Y)
            elif method == 'KMeans':
                clustering = getattr(cluster, method)(n_clusters = n_cluster, n_init = ite_num).fit(Y)
            else:
                clustering = getattr(cluster, method[tt])(n_clusters = n_cluster[i]).fit(Y)
            for j in np.nditer(np.unique(clustering.labels_)):
                idx =  (clustering.labels_ == j)   
                candi_matrix[idx, :, i, tt] = np.logical_and(Y[idx,:] ==0, np.mean(Y[idx, :] == 0, 0) >= threshold)
            candi_matrix[idx_non_zero, i, tt] = 1 
            
    idx_matrix[np.sum(candi_matrix, axis=(3,2))/(K1*K2) >= 0.5] = 1  # Combinine different results by the major voting.
    
    return idx_matrix


def get_ARI(Y, true_labels, n_cluster=2, method="KMeans"):
    
    """
    A function for obtaining ARI of the clustering results on the low-dimensional data
    Y- data array of size n samples by n features. 
    """

    ite_num = 30
    pred_labels = getattr(cluster, method)(n_clusters = n_cluster, n_init = ite_num).fit(Y).labels_
        
  
    return adjusted_rand_score(true_labels, pred_labels)






