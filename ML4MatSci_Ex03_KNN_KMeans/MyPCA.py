import numpy as np
import matplotlib.pyplot as plt



class MyPCA:
    """
    Creates a PCA transform instance. 
    
    Inputs: 
    - n_components: number of PCA components
    
    Output: 
    - Object that implements PCA with the given number of components
      a given dataset.
      
    Example Usage: 
    
    mypca = MyPCA(10)
    mypca.fit(X, mu, std) #we provide mu and std to enforce mypca to use the given statistics
    z = mypca.transform(X)
    reconst = mypca.inverse_transform(z)
    total_expl_variance_ratio = np.sum(mypca.explained_variance())
    print("total explained variance ratio: ", total_expl_variance_ratio.round(4))
    """
    
    
    
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, X, mu = None, std = None):
        """
        Creates a PCA transform by computing mean and standard deviation, or by using
        user-defined mean and standard deviation. 
        
        Adds the attributes: mean, std, eigvals and B_matrix to the PCA object. These
        can be then used to apply PCA on a given dataset. 
        
        Inputs:
        - X : Dataset on which dimension reduction by PCA is to be applied. 
        - mu (optional): User-defined mean for normalizing data.  
        - std (optional): User-defined standard deviation for normalizing data. 
        """
        
        # Make a copy of the data
        X = X.copy()
        #print("X shape:", X.shape)
        
        # Calculate mean and standard deviation
        if mu is not None:
            self.mean = mu
        else:
            self.mean = np.mean(X, axis=0)
     
            
        if std is not None:
            self.std = std
            self.std[self.std == 0] = 1 # Replace zero std deviation with 1
        else:
            self.std = np.std(X, axis=0)
            self.std[self.std == 0] = 1 # Replace zero std deviation with 1
        
        # Standardize the data
        X_std = (X - self.mean) / self.std
        
        # Calculate covariance matrix
        S = np.cov(X_std, rowvar=False)
        
        # Eigendecomposition
        eig_vals, eig_vecs = np.linalg.eig(S)
        
        # Ensure eigenvalues and eigenvectors are real
        eig_vals = np.real(eig_vals)
        eig_vecs = np.real(eig_vecs)
        
        # Sort eigenvalues and eigenvectors
        k = np.argsort(eig_vals)[::-1]
        self.eig_vals = eig_vals[k]
        eig_vecs = eig_vecs[:, k]
        
        # Select top n_components eigenvectors
        self.B_matrix = eig_vecs[:, :self.n_components]
        #print("B matrix shape:", self.B_matrix.shape)
        
        return self
    
    def transform(self, X):
        """
        Applies PCA to the given dataset. 
        
        Inputs:
        - X: Dataset matrix on which PCA is applied. Expected dimensions, (n_dimensions, n_samples). 
          n_dimensions refers to the dimensionality of a sample. 
          
        Output: 
        - Z : Dataset with reduced number of dimensions. Expected dimension: (n_pca_components, n_samples). 
        """
        
        
        # Make a copy of the data
        X = X.copy()
        
        # Standardize the data
        X_std = (X - self.mean) / self.std
        #print("X std shape:", X_std.shape)
        
        # Project the data onto the new space
        Z = np.dot(X_std, self.B_matrix)
        
        return Z
    
    def inverse_transform(self, X):
        """
        Inverts the PCA applied on the dataset, to reconstruct the sample in original dimension space. 
        
        Input: 
        - X : Dataset matrix on which PCA is applied. Expected dimensions, (n_dimensions, n_samples). 
          n_dimensions refers to the dimensionality of a sample. 
          
        Output: 
        - reconst: Dataset matrix with original dimensions of the samples after inversing the PCA transform. 
          Expected dimensions, (n_dimensions, n_samples), where n_dimensions refers to the dimensionality 
          of the original sample. 
        
        """
        
        
        
        # Project back to original space
        X_original_space = np.dot(X, self.B_matrix.T)
        
        # Reverse the standardization
        reconst = X_original_space * self.std + self.mean
        
        # Ensure the result is real
        reconst = np.real(reconst)
        
        return reconst
    
    def explained_variance(self):
        """
        Computes the explained variance of the PCA transform. 
        
        """
        
        
        
        # Total variance is the sum of eigenvalues
        total_variance = np.sum(self.eig_vals)

        # Explained variance for each component
        explained_variance = self.eig_vals[:self.n_components] / total_variance
        
        return explained_variance
    
    
    
