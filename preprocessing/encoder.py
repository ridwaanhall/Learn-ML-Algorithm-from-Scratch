"""
Encoding utilities for categorical data
"""
import numpy as np
from typing import List, Optional, Dict, Any


class LabelEncoder:
    """
    Encode target labels with value between 0 and n_classes-1
    """
    
    def __init__(self):
        """Initialize LabelEncoder"""
        self.classes_ = None
        self.is_fitted = False
    
    def fit(self, y: np.ndarray) -> 'LabelEncoder':
        """
        Fit label encoder
        
        Args:
            y: Target values of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        self.classes_ = np.unique(y)
        self.is_fitted = True
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform labels to normalized encoding
        
        Args:
            y: Target values to encode
            
        Returns:
            Encoded labels
        """
        if not self.is_fitted:
            raise ValueError("LabelEncoder has not been fitted yet. Call fit() first.")
        
        encoded = np.zeros(len(y), dtype=int)
        for i, label in enumerate(y):
            if label not in self.classes_:
                raise ValueError(f"Unknown label: {label}")
            encoded[i] = np.where(self.classes_ == label)[0][0]
        
        return encoded
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Fit label encoder and return encoded labels
        
        Args:
            y: Target values
            
        Returns:
            Encoded labels
        """
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform labels back to original encoding
        
        Args:
            y: Encoded labels
            
        Returns:
            Original labels
        """
        if not self.is_fitted:
            raise ValueError("LabelEncoder has not been fitted yet. Call fit() first.")
        
        return self.classes_[y]


class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array
    """
    
    def __init__(self, sparse: bool = False, drop: Optional[str] = None):
        """
        Initialize OneHotEncoder
        
        Args:
            sparse: Will return sparse matrix if set True (not implemented)
            drop: Specifies a methodology to drop one category per feature
        """
        self.sparse = sparse
        self.drop = drop
        self.categories_ = None
        self.n_features_in_ = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """
        Fit OneHotEncoder to X
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_in_ = X.shape[1]
        self.categories_ = []
        
        for i in range(X.shape[1]):
            unique_values = np.unique(X[:, i])
            self.categories_.append(unique_values)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X using one-hot encoding
        
        Args:
            X: Input data to transform
            
        Returns:
            One-hot encoded array
        """
        if not self.is_fitted:
            raise ValueError("OneHotEncoder has not been fitted yet. Call fit() first.")
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but encoder was fitted with {self.n_features_in_}")
        
        encoded_columns = []
        
        for i in range(X.shape[1]):
            feature_categories = self.categories_[i]
            n_categories = len(feature_categories)
            
            # Create one-hot encoding for this feature
            feature_encoded = np.zeros((X.shape[0], n_categories))
            
            for j, category in enumerate(feature_categories):
                mask = X[:, i] == category
                feature_encoded[mask, j] = 1
            
            # Handle drop option
            if self.drop == 'first' and n_categories > 1:
                feature_encoded = feature_encoded[:, 1:]
            
            encoded_columns.append(feature_encoded)
        
        return np.concatenate(encoded_columns, axis=1)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit OneHotEncoder to X, then transform X
        
        Args:
            X: Input data
            
        Returns:
            One-hot encoded array
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convert the one-hot encoded data back to the original representation
        
        Args:
            X: One-hot encoded data
            
        Returns:
            Original categorical data
        """
        if not self.is_fitted:
            raise ValueError("OneHotEncoder has not been fitted yet. Call fit() first.")
        
        original_data = []
        col_idx = 0
        
        for i, feature_categories in enumerate(self.categories_):
            n_categories = len(feature_categories)
            
            # Adjust for dropped categories
            if self.drop == 'first' and n_categories > 1:
                n_categories_encoded = n_categories - 1
                # Add back the dropped category
                feature_encoded = np.zeros((X.shape[0], n_categories))
                feature_encoded[:, 0] = 1 - np.sum(X[:, col_idx:col_idx + n_categories_encoded], axis=1)
                feature_encoded[:, 1:] = X[:, col_idx:col_idx + n_categories_encoded]
            else:
                n_categories_encoded = n_categories
                feature_encoded = X[:, col_idx:col_idx + n_categories_encoded]
            
            # Convert one-hot back to categorical
            feature_original = np.zeros(X.shape[0], dtype=feature_categories.dtype)
            for j, category in enumerate(feature_categories):
                mask = feature_encoded[:, j] == 1
                feature_original[mask] = category
            
            original_data.append(feature_original)
            col_idx += n_categories_encoded
        
        return np.column_stack(original_data)
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation
        
        Args:
            input_features: Input feature names
            
        Returns:
            Output feature names
        """
        if not self.is_fitted:
            raise ValueError("OneHotEncoder has not been fitted yet. Call fit() first.")
        
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        
        feature_names = []
        
        for i, feature_categories in enumerate(self.categories_):
            feature_name = input_features[i]
            
            categories_to_include = feature_categories
            if self.drop == 'first' and len(feature_categories) > 1:
                categories_to_include = feature_categories[1:]
            
            for category in categories_to_include:
                feature_names.append(f"{feature_name}_{category}")
        
        return feature_names


class OrdinalEncoder:
    """
    Encode categorical features as an integer array
    """
    
    def __init__(self):
        """Initialize OrdinalEncoder"""
        self.categories_ = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'OrdinalEncoder':
        """
        Fit the OrdinalEncoder to X
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        self.categories_ = []
        for i in range(X.shape[1]):
            unique_values = np.unique(X[:, i])
            self.categories_.append(unique_values)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X to ordinal codes
        
        Args:
            X: Input data to transform
            
        Returns:
            Ordinal encoded array
        """
        if not self.is_fitted:
            raise ValueError("OrdinalEncoder has not been fitted yet. Call fit() first.")
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        encoded = np.zeros(X.shape, dtype=int)
        
        for i in range(X.shape[1]):
            feature_categories = self.categories_[i]
            for j, value in enumerate(X[:, i]):
                if value in feature_categories:
                    encoded[j, i] = np.where(feature_categories == value)[0][0]
                else:
                    raise ValueError(f"Unknown category: {value}")
        
        return encoded
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit OrdinalEncoder to X, then transform X
        
        Args:
            X: Input data
            
        Returns:
            Ordinal encoded array
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convert the ordinal codes back to the original representation
        
        Args:
            X: Ordinal encoded data
            
        Returns:
            Original categorical data
        """
        if not self.is_fitted:
            raise ValueError("OrdinalEncoder has not been fitted yet. Call fit() first.")
        
        original = np.zeros(X.shape, dtype=object)
        
        for i in range(X.shape[1]):
            feature_categories = self.categories_[i]
            for j in range(X.shape[0]):
                original[j, i] = feature_categories[X[j, i]]
        
        return original
