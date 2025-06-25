"""
Cross Entropy loss function for classification tasks
"""
import numpy as np
from typing import Tuple


class CrossEntropyLoss:
    """
    Cross Entropy loss function for classification tasks
    
    For binary classification: -[y*log(p) + (1-y)*log(1-p)]
    For multiclass: -Σ(y_i * log(p_i))
    """
    
    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
        """
        Compute binary cross entropy loss
        
        Args:
            y_true: True binary labels of shape (n_samples,)
            y_pred: Predicted probabilities of shape (n_samples,)
            epsilon: Small value to prevent log(0)
            
        Returns:
            Binary cross entropy loss
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
        """
        Compute categorical cross entropy loss
        
        Args:
            y_true: True one-hot encoded labels of shape (n_samples, n_classes)
            y_pred: Predicted probabilities of shape (n_samples, n_classes)
            epsilon: Small value to prevent log(0)
            
        Returns:
            Categorical cross entropy loss
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def sparse_categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
        """
        Compute sparse categorical cross entropy loss
        
        Args:
            y_true: True class indices of shape (n_samples,)
            y_pred: Predicted probabilities of shape (n_samples, n_classes)
            epsilon: Small value to prevent log(0)
            
        Returns:
            Sparse categorical cross entropy loss
        """
        if len(y_true) != y_pred.shape[0]:
            raise ValueError("Number of samples in y_true and y_pred must match")
        
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Extract probabilities for true classes
        true_class_probs = y_pred[np.arange(len(y_true)), y_true.astype(int)]
        
        return -np.mean(np.log(true_class_probs))
    
    @staticmethod
    def binary_cross_entropy_gradient(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
        """
        Compute gradient of binary cross entropy loss
        
        Args:
            y_true: True binary labels of shape (n_samples,)
            y_pred: Predicted probabilities of shape (n_samples,)
            epsilon: Small value to prevent division by 0
            
        Returns:
            Gradient of shape (n_samples,)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        # Clip predictions to prevent division by 0
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        n = len(y_true)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n
    
    @staticmethod
    def categorical_cross_entropy_gradient(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
        """
        Compute gradient of categorical cross entropy loss
        
        Args:
            y_true: True one-hot encoded labels of shape (n_samples, n_classes)
            y_pred: Predicted probabilities of shape (n_samples, n_classes)
            epsilon: Small value to prevent division by 0
            
        Returns:
            Gradient of shape (n_samples, n_classes)
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        
        # Clip predictions to prevent division by 0
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        n = y_true.shape[0]
        return -y_true / y_pred / n
