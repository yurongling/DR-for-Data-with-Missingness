import numpy as np

METHODS = ['uniform']

def generate_missing_observations(X, method = 'uniform', low_i=0.1, high_i=0.5, low_j=0.1, high_j=0.5, loc_i=0, sigma_i=1, loc_j=0,sigma_j=1):
    """
    Generate missing observations for complete data matrix X with two uniform distributions.
    
    Params:
    X, ndarray, (n,d), complete data matrix.
    method: str, specify the missing mechanism.
    low_i, high_i, low_j, high_j: params of uniform distributions.
    loc_i,loc_j,sigma_i,sigma_j: params of Gaussian distributions.
    
    Return:
    X_miss, ndarray, (n,d), data matrix with missing observations represented by np.nan
    """
    assert method in METHODS , "Missing mechanism not supported"
    n,d = X.shape
    if method == 'uniform':
        p_i = np.random.uniform(low_i,high_i,n).reshape(-1,1)
        p_j = np.random.uniform(low_j,high_j,d).reshape(-1,1)

    
    p = p_i.dot(p_j.T) # missing probabilities

    mask = np.random.binomial(n=1,p=p,size=X.shape).astype(bool) #True for missing values, False for others

    X_miss = np.copy(X)

    X_miss[mask] = np.nan

    print('Fraction of missing data: %2.3f' % mask.mean())
    
    return X_miss

def generate_simulated_data(n_clusters, n, d, k, sigma):
    """
    Generates data with multiple clusters. Data are generated in a PPCA setting, 
    which can be treated as the GPLVM model with a linear kernel.
    
    """
    mu = 3
    range_from_value = .1

    if n_clusters == 1:
        Z = np.random.multivariate_normal(mean=np.zeros([k, ]), cov=np.eye(k), size=n).transpose()
        cluster_ids = np.ones([n])

    else:
        Z = np.zeros([k, n])
        cluster_ids = np.array([np.random.choice(range(n_clusters)) for i in range(n)])

        for id in list(set(cluster_ids)):
            idxs = cluster_ids == id
            cluster_mu = (np.random.random([k]) - .5) * 5
            Z[:, idxs] = np.random.multivariate_normal(mean=cluster_mu, cov=.01 * np.eye(k), size=idxs.sum()).transpose()

    A = np.random.random([d, k]) - .5
    mu = np.array([(np.random.uniform() * range_from_value * 2 + (1 - range_from_value)) * mu for i in range(d)])
    sigmas = np.array([(np.random.uniform() * range_from_value * 2 + (1 - range_from_value)) * sigma for i in range(d)])
    noise = np.zeros([d, n])

    for j in range(d):
        noise[j, :] = mu[j] + np.random.normal(loc=0, scale=sigmas[j], size=n)

    X = (np.dot(A, Z) + noise).transpose()


    return X, Z.transpose(), cluster_ids
