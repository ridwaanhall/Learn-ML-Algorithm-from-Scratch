"""
Label Encoder for categorical data preprocessing.

This module provides encoding utilities for converting categorical variables
into numerical format suitable for machine learning algorithms.
"""
import numpy as np

class LabelEncoder:
    """
    Encode categorical labels as integers.
    
    This transformer converts categorical labels into integer labels
    in the range [0, n_classes-1].
    """
    
    def __init__(self):
        """Initialize label encoder."""
        self.classes_ = None
        self.class_to_index_ = None
        self.index_to_class_ = None
    
    def fit(self, y):
        """
        Fit label encoder to categorical labels.
        
        Args:
            y (array-like): Target labels
            
        Returns:
            self: Returns the instance itself
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.class_to_index_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.index_to_class_ = {idx: cls for idx, cls in enumerate(self.classes_)}
        return self
    
    def transform(self, y):
        """
        Transform categorical labels to integer labels.
        
        Args:
            y (array-like): Target labels to transform
            
        Returns:
            np.ndarray: Encoded labels
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder must be fitted before transforming")
        
        y = np.asarray(y)
        encoded = np.zeros(len(y), dtype=int)
        
        for i, label in enumerate(y):
            if label not in self.class_to_index_:
                raise ValueError(f"Label '{label}' not seen during fit")
            encoded[i] = self.class_to_index_[label]
        
        return encoded
    
    def fit_transform(self, y):
        """
        Fit label encoder and transform categorical labels.
        
        Args:
            y (array-like): Target labels
            
        Returns:
            np.ndarray: Encoded labels
        """
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        """
        Transform integer labels back to original categorical labels.
        
        Args:
            y (array-like): Integer labels to transform back
            
        Returns:
            np.ndarray: Original categorical labels
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder must be fitted before inverse transforming")
        
        y = np.asarray(y)
        decoded = np.zeros(len(y), dtype=object)
        
        for i, index in enumerate(y):
            if index not in self.index_to_class_:
                raise ValueError(f"Index '{index}' not valid")
            decoded[i] = self.index_to_class_[index]
        
        return decoded

class OneHotEncoder:
    """
    Encode categorical features as one-hot numeric arrays.
    
    This transformer converts categorical features into a binary matrix
    where each column represents one category.
    """
    
    def __init__(self, sparse=False):
        """
        Initialize one-hot encoder.
        
        Args:
            sparse (bool): Whether to return sparse matrix (not implemented)
        """
        self.sparse = sparse
        self.categories_ = None
        self.n_categories_ = None
    
    def fit(self, X):
        """
        Fit one-hot encoder to categorical features.
        
        Args:
            X (array-like): Input features
            
        Returns:
            self: Returns the instance itself
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.categories_ = []
        self.n_categories_ = []
        
        for col in range(X.shape[1]):
            unique_values = np.unique(X[:, col])
            self.categories_.append(unique_values)
            self.n_categories_.append(len(unique_values))
        
        return self
    
    def transform(self, X):
        """
        Transform categorical features to one-hot encoding.
        
        Args:
            X (array-like): Input features to transform
            
        Returns:
            np.ndarray: One-hot encoded features
        """
        if self.categories_ is None:
            raise ValueError("OneHotEncoder must be fitted before transforming")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        total_categories = sum(self.n_categories_)
        
        encoded = np.zeros((n_samples, total_categories))
        col_idx = 0
        
        for feature_idx in range(X.shape[1]):
            categories = self.categories_[feature_idx]
            n_cats = self.n_categories_[feature_idx]
            
            for sample_idx in range(n_samples):
                value = X[sample_idx, feature_idx]
                if value in categories:
                    cat_idx = np.where(categories == value)[0][0]
                    encoded[sample_idx, col_idx + cat_idx] = 1
                else:
                    raise ValueError(f"Value '{value}' not seen during fit")
            
            col_idx += n_cats
        
        return encoded
    
    def fit_transform(self, X):
        """
        Fit one-hot encoder and transform categorical features.
        
        Args:
            X (array-like): Input features
            
        Returns:
            np.ndarray: One-hot encoded features
        """
        return self.fit(X).transform(X)
