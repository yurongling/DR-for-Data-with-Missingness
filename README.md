# Dimension Reduction for Data with Heterogeneous Missingness

If you are using count data, we recommend taking the log (ie, Y = log2(1 + count data)) prior to using the proposed the bias correction.

Functions for correcting the bias are contained in the bias_correction.py. gplvm_gram.py is for running the bias-corrected GPLVM, which is modified from the GPflow package v1.3.0.


## Sample usage
### To get the indicator array of n_samples by n_features with integer 0 indicating dropouts for scRNA-seq data:

n_cluster = np.array([4,6,8,10,12]).astype(int)

M = id_consensus(df, n_cluster, 0.85, ['KMeans', 'SpectralClustering']) 


### To get the bias-corrected Gram matrix:

BC_G = BC_Gram(df,M)


### To get the  k-dimensional components from bias-corrected PCA:

BC_PCA_x = BC_mdsReduce(df,M = M,k=k) #BC-PCA, where 0 in M denotes the missing observation.


### To get the k-dimensional components from bias-corrected tSNE:

BC_df= BC_mdsReduce(df,M,'all') # representation of data obtained from the bias-corrected PCA with dimension automatically determined. 

BC_tSNE_x = TSNE(n_components= k).fit_transform(BC_df) 


### To get the k-dimensional data from bias-corrected UMAP:

BC_UMAP_x = umap.UMAP(n_components = k).fit_transform(BC_df) # BC-UMAP


### For using the bias-corrected GPLVM, please add gplvm_gram.py in the gpflow package (model folder) v1.3.0. The sample use of getting k-dimensional data is as follows:
Gram = BC_Gram(df,M)

bc_gplvm = gpflow.models.GPLVM_Gram( Y=df, latent_dim = 2, Gram= Gram)
opt = gpflow.train.ScipyOptimizer()
opt.minimize(bc_gplvm, maxiter=1000)  
BC_GPLVM_x = bc_gplvm.X.value #BC-GPLVM
