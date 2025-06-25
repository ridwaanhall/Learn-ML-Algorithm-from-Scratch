"""
Data scaling and normalization utilities
"""
import numpy as np
from typing import Optional, Tuple


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance
    
    The standard score of a sample x is calculated as:
    z = (x - u) / s
    where u is the mean and s is the standard deviation
    """
    
    def __init__(self):
        """Initialize StandardScaler"""
        self.mean_ = None
        self.std_ = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute the mean and std to be used for later scaling
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features using the fitted parameters
        
        Args:
            X: Data to scale of shape (n_samples, n_features)
            
        Returns:
            Scaled data
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        
        if X.shape[1] != len(self.mean_):
            raise ValueError("X has different number of features than fitted data")
        
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it
        
        Args:
            X: Data to fit and transform
            
        Returns:
            Scaled data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale back the data to the original representation
        
        Args:
            X: Scaled data
            
        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """
    Transform features by scaling each feature to a given range
    
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        """
        Initialize MinMaxScaler
        
        Args:
            feature_range: Desired range of transformed data
        """
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Compute the minimum and maximum to be used for later scaling
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        
        data_range = self.data_max_ - self.data_min_
        # Avoid division by zero
        data_range[data_range == 0] = 1.0
        
        scale = (self.feature_range[1] - self.feature_range[0]) / data_range
        min_val = self.feature_range[0] - self.data_min_ * scale
        
        self.scale_ = scale
        self.min_ = min_val
        self.is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features using the fitted parameters
        
        Args:
            X: Data to scale of shape (n_samples, n_features)
            
        Returns:
            Scaled data
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        
        if X.shape[1] != len(self.scale_):
            raise ValueError("X has different number of features than fitted data")
        
        return X * self.scale_ + self.min_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it
        
        Args:
            X: Data to fit and transform
            
        Returns:
            Scaled data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Undo the scaling of X according to feature_range
        
        Args:
            X: Scaled data
            
        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        
        return (X - self.min_) / self.scale_


class RobustScaler:
    """
    Scale features using statistics that are robust to outliers
    
    This Scaler removes the median and scales the data according to
    the quantile range (defaults to IQR: Interquartile Range).
    """
    
    def __init__(self, quantile_range: Tuple[float, float] = (25.0, 75.0)):
        """
        Initialize RobustScaler
        
        Args:
            quantile_range: Quantile range used to calculate scale
        """
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'RobustScaler':
        """
        Compute the median and quantiles to be used for later scaling
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        
        self.center_ = np.median(X, axis=0)
        
        quantiles = np.percentile(X, self.quantile_range, axis=0)
        scale = quantiles[1] - quantiles[0]
        
        # Avoid division by zero
        scale[scale == 0] = 1.0
        
        self.scale_ = scale
        self.is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Center and scale the data
        
        Args:
            X: Data to scale of shape (n_samples, n_features)
            
        Returns:
            Scaled data
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        
        if X.shape[1] != len(self.center_):
            raise ValueError("X has different number of features than fitted data")
        
        return (X - self.center_) / self.scale_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it
        
        Args:
            X: Data to fit and transform
            
        Returns:
            Scaled data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale back the data to the original representation
        
        Args:
            X: Scaled data
            
        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        
        return X * self.scale_ + self.center_
