"""
Principal Component Analysis (PCA) implementation from scratch
"""
import numpy as np
from .base_model import BaseUnsupervised
from typing import Optional, Union


class PCA(BaseUnsupervised):
    """
    Principal Component Analysis (PCA)
    
    Linear dimensionality reduction using Singular Value Decomposition (SVD)
    to project data to a lower dimensional space
    """
    
    def __init__(self, n_components: Optional[Union[int, float]] = None, 
                 whiten: bool = False, random_state: Optional[int] = None):
        """
        Initialize PCA
        
        Args:
            n_components: Number of components to keep
                         If int: exact number of components
                         If float: proportion of variance to retain
                         If None: keep all components
            whiten: Whether to whiten the components
            random_state: Random seed (for future extensions)
        """
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        
        # Fitted attributes
        self.components_ = None  # Principal axes in feature space
        self.explained_variance_ = None  # Variance explained by each component
        self.explained_variance_ratio_ = None  # Ratio of variance explained
        self.singular_values_ = None  # Singular values
        self.mean_ = None  # Per-feature empirical mean
        self.n_components_ = None  # Actual number of components
        self.noise_variance_ = None  # Noise variance
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PCA':
        """
        Fit PCA model
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Ignored (for API compatibility)
            
        Returns:
            Self for method chaining
        """
        self._validate_input(X)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Perform SVD
        # X_centered = U * S * V^T
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Components are the rows of V^T (columns of V)
        self.components_ = Vt
        
        # Explained variance
        self.explained_variance_ = (s ** 2) / (n_samples - 1)
        
        # Total variance
        total_variance = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        # Singular values
        self.singular_values_ = s
        
        # Determine number of components to keep
        if self.n_components is None:
            self.n_components_ = min(n_samples, n_features)
        elif isinstance(self.n_components, int):
            if self.n_components <= 0:
                raise ValueError("n_components must be positive")
            self.n_components_ = min(self.n_components, min(n_samples, n_features))
        elif isinstance(self.n_components, float):
            if not 0 < self.n_components <= 1:
                raise ValueError("n_components as float must be between 0 and 1")
            
            # Find number of components that explain desired variance
            cumsum_var_ratio = np.cumsum(self.explained_variance_ratio_)
            self.n_components_ = np.searchsorted(cumsum_var_ratio, self.n_components) + 1
            self.n_components_ = min(self.n_components_, min(n_samples, n_features))
        else:
            raise ValueError("n_components must be int, float, or None")
        
        # Keep only the required number of components
        self.components_ = self.components_[:self.n_components_]
        self.explained_variance_ = self.explained_variance_[:self.n_components_]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components_]
        self.singular_values_ = self.singular_values_[:self.n_components_]
        
        # Calculate noise variance (average of remaining eigenvalues)
        if self.n_components_ < min(n_samples, n_features):
            remaining_variance = np.sum(self.explained_variance_[self.n_components_:])
            remaining_components = min(n_samples, n_features) - self.n_components_
            self.noise_variance_ = remaining_variance / remaining_components if remaining_components > 0 else 0
        else:
            self.noise_variance_ = 0
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X
        
        Args:
            X: Data to transform of shape (n_samples, n_features)
            
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        self._check_fitted()
        self._validate_input(X)
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features, but PCA was fitted with {self.n_features}")
        
        # Center the data
        X_centered = X - self.mean_
        
        # Project onto principal components
        X_transformed = np.dot(X_centered, self.components_.T)
        
        # Apply whitening if requested
        if self.whiten:
            X_transformed = X_transformed / np.sqrt(self.explained_variance_)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit PCA and apply dimensionality reduction
        
        Args:
            X: Data to fit and transform
            y: Ignored (for API compatibility)
            
        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space
        
        Args:
            X: Transformed data of shape (n_samples, n_components)
            
        Returns:
            Data in original space of shape (n_samples, n_features)
        """
        self._check_fitted()
        
        if X.shape[1] != self.n_components_:
            raise ValueError(f"X has {X.shape[1]} components, but PCA has {self.n_components_}")
        
        # Reverse whitening if it was applied
        if self.whiten:
            X = X * np.sqrt(self.explained_variance_)
        
        # Project back to original space
        X_original = np.dot(X, self.components_) + self.mean_
        
        return X_original
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction (same as transform)
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        return self.transform(X)
    
    def get_covariance(self) -> np.ndarray:
        """
        Compute data covariance with the generative model
        
        Returns:
            Estimated covariance matrix
        """
        self._check_fitted()
        
        components = self.components_
        exp_var = self.explained_variance_
        
        if self.whiten:
            components = components * np.sqrt(exp_var[:, np.newaxis])
        
        exp_var_diff = np.maximum(exp_var - self.noise_variance_, 0.)
        cov = np.dot(components.T * exp_var_diff, components)
        cov.flat[::len(cov) + 1] += self.noise_variance_  # Add noise to diagonal
        
        return cov
    
    def get_precision(self) -> np.ndarray:
        """
        Compute data precision matrix with the generative model
        
        Returns:
            Estimated precision matrix
        """
        self._check_fitted()
        
        n_features = self.components_.shape[1]
        
        # Use the matrix inversion lemma for efficiency
        if self.n_components_ == 0:
            return np.eye(n_features) / self.noise_variance_
        
        if self.n_components_ == n_features:
            return np.linalg.inv(self.get_covariance())
        
        # Sherman-Morrison-Woodbury formula
        components = self.components_
        exp_var = self.explained_variance_
        exp_var_diff = np.maximum(exp_var - self.noise_variance_, 0.)
        
        precision = np.eye(n_features) / self.noise_variance_
        precision -= np.dot(
            components.T / self.noise_variance_,
            np.dot(
                np.linalg.inv(np.diag(1. / exp_var_diff) + 
                            np.dot(components / self.noise_variance_, components.T)),
                components / self.noise_variance_
            )
        )
        
        return precision
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood of each sample under the model
        
        Args:
            X: Data samples
            
        Returns:
            Log-likelihood of each sample
        """
        self._check_fitted()
        self._validate_input(X)
        
        precision = self.get_precision()
        X_centered = X - self.mean_
        
        # Compute log-likelihood
        log_like = -0.5 * np.sum(X_centered * np.dot(X_centered, precision), axis=1)
        log_like -= 0.5 * (self.n_features * np.log(2 * np.pi) - 
                          np.linalg.slogdet(precision)[1])
        
        return log_like
    
    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """
        Return the average log-likelihood of all samples
        
        Args:
            X: Data samples
            y: Ignored (for API compatibility)
            
        Returns:
            Average log-likelihood
        """
        return np.mean(self.score_samples(X))
