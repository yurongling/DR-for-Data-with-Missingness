import tensorflow as tf
import numpy as np
from .. import likelihoods
from .. import settings

from .. import transforms
from .. import kernels
from .. import features

from ..params import Parameter
from ..mean_functions import Zero
from ..expectations import expectation
from ..probability_distributions import DiagonalGaussian
from ..logdensities import multivariate_normal

from ..conditionals import base_conditional
from ..params import DataHolder
from ..decors import params_as_tensors
from ..decors import name_scope
from ..logdensities import multivariate_normal

from .model import GPModel
from sklearn import cluster
from sklearn import preprocessing



float_type = settings.float_type


class GPLVM_Gram(GPModel):
    """
    Standard GPLVM where the likelihood can be optimised with respect to the latent X.
    """

    def __init__(self, Y, latent_dim,Gram, X_mean=None, kern=None, mean_function=None, **kwargs):
        """
        Initialise GPLVM object. This method only works with a Gaussian likelihood.

        :param Y: data matrix, size N (number of points) x D (dimensions)
        :param Z: matrix of inducing points, size M (inducing points) x Q (latent dimensions)
        :param X_mean: latent positions (N x Q), for the initialisation of the latent space.
        :param kern: kernel specification, by default RBF
        :param mean_function: mean function, by default None.
        """
    
        if mean_function is None:
            mean_function = Zero()
        if kern is None:
            kern = kernels.RBF(latent_dim, ARD=False)
        if X_mean is None:
            #X_mean = PCA_reduce(Y, latent_dim)
            X_mean =mdsReduce(Y,latent_dim)
        num_latent = X_mean.shape[1]
        if num_latent != latent_dim:
            msg = 'Passed in number of latent {0} does not match initial X {1}.'
            raise ValueError(msg.format(latent_dim, num_latent))
        if Y.shape[1] < num_latent:
            raise ValueError('More latent dimensions than observed.')
         
        Gram = Gram

        likelihood = likelihoods.Gaussian()
        X = DataHolder(X_mean)
        Y = DataHolder(Y)

        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        
        self.num_samples, self.num_dim = Y.shape
        self.Gram = Gram

        del self.X  # in GPLVM this is a Param
        self.X = Parameter(X_mean)
        

    @name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        r"""
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
        L = tf.cholesky(K)
        #m = self.mean_function(self.X)
        #p = multivariate_normal(self.Y, m, L) 
        
        #m = self.mean_function(self.X)
        L = tf.cholesky(K)
        #K_inv = tf.matrix_inverse(L)
        #K_inv = tf.matmul(tf.transpose(K_inv), K_inv)
                
        alpha2 = tf.matrix_solve(K,self.Gram)

        logpdf = -0.5*self.num_samples * np.log(2 * np.pi)
        logpdf -= tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
        logpdf -= 0.5* tf.reduce_sum(tf.matrix_diag_part(alpha2))
        #logpdf -= 0.5* tf.reduce_sum(tf.matrix_diag_part(tf.matmul(K_inv, self.Gram)))
        
        #alpha = tf.matrix_triangular_solve(L, self.Y, lower=True)
        #p = - 0.5 * tf.reduce_sum(tf.square(alpha), 0)
        #p -= 0.5 * self.num_samples * np.log(2 * np.pi)
        #p -= tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))

        return tf.reduce_sum(logpdf)

    @name_scope('predict')
    @params_as_tensors   
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, the points at which we want to predict.

        This method computes

            p(F* | Y)

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        y = self.Y - self.mean_function(self.X)
        Kmn = self.kern.K(self.X, Xnew)
        Kmm_sigma = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
        Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)
        f_mean, f_var = base_conditional(Kmn, Kmm_sigma, Knn, y, full_cov=full_cov, white=False)  # N x P, N x P or P x N x N
        return f_mean + self.mean_function(Xnew), f_var
        


def PCA_reduce(X, Q):
    """
    A helpful function for linearly reducing the dimensionality of the data X
    to Q.
    :param X: data array of size N (number of points) x D (dimensions)
    :param Q: Number of latent dimensions, Q < D
    :return: PCA projection array of size N x Q.
    """
    assert Q <= X.shape[1], 'Cannot have more latent dimensions than observed'
    evals, evecs = np.linalg.eigh(np.cov(X.T))
    W = evecs[:, -Q:]
    return (X - X.mean(0)).dot(W)

def mdsReduce(Y,k):
    Yn =  Y - np.mean(Y, axis=0)
    G = Yn.dot(Yn.T)

    evalu,evect = np.linalg.eigh(G)
    i = np.argsort(evalu)[::-1]
    E = evalu[i]
    E = np.diag(E[:k])
    W = evect[:,i]
    W = W[:,:k]
    return W.dot(np.sqrt(E))
   


