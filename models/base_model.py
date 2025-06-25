"""
Base classes for ML models
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Optional


class BaseModel(ABC):
    """
    Abstract base class for all ML models
    
    This class defines the interface that all ML models must implement.
    It follows the scikit-learn API pattern for consistency.
    """
    
    def __init__(self):
        """Initialize base model"""
        self.is_fitted = False
        self.feature_names = None
        self.n_features = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Fit the model to training data
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        pass
    
    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Raises:
            ValueError: If input validation fails
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise ValueError("y must be a numpy array")
            
            if len(y.shape) != 1:
                raise ValueError("y must be a 1D array")
            
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have the same number of samples")
    
    def _check_fitted(self) -> None:
        """
        Check if model has been fitted
        
        Raises:
            ValueError: If model has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")


class BaseRegressor(BaseModel):
    """
    Base class for regression models
    """
    
    def __init__(self):
        super().__init__()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction
        
        Args:
            X: Test features
            y: True values
            
        Returns:
            R^2 score
        """
        from ..metrics.regression import RegressionMetrics
        y_pred = self.predict(X)
        return RegressionMetrics.r2_score(y, y_pred)


class BaseClassifier(BaseModel):
    """
    Base class for classification models
    """
    
    def __init__(self):
        super().__init__()
        self.classes_ = None
        self.n_classes = None
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Accuracy score
        """
        from ..metrics.classification import ClassificationMetrics
        y_pred = self.predict(X)
        return ClassificationMetrics.accuracy(y, y_pred)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        # Default implementation returns hard predictions as probabilities
        # Subclasses should override this method for proper probability estimates
        predictions = self.predict(X)
        n_samples = len(predictions)
        n_classes = len(self.classes_)
        
        probabilities = np.zeros((n_samples, n_classes))
        for i, pred in enumerate(predictions):
            class_idx = np.where(self.classes_ == pred)[0][0]
            probabilities[i, class_idx] = 1.0
            
        return probabilities


class BaseUnsupervised(BaseModel):
    """
    Base class for unsupervised learning models
    """
    
    def __init__(self):
        super().__init__()
        self.labels_ = None
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and return cluster labels or transformed data
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Labels or transformed data
        """
        self.fit(X)
        return self.predict(X)
