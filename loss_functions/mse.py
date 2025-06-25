"""
Mean Squared Error (MSE) loss function
"""
import numpy as np
from typing import Tuple


class MSELoss:
    """
    Mean Squared Error loss function for regression tasks
    
    MSE = (1/n) * Σ(y_true - y_pred)²
    """
    
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute MSE loss
        
        Args:
            y_true: True target values of shape (n_samples,)
            y_pred: Predicted values of shape (n_samples,)
            
        Returns:
            MSE loss value
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradient of MSE loss with respect to predictions
        
        Args:
            y_true: True target values of shape (n_samples,)
            y_pred: Predicted values of shape (n_samples,)
            
        Returns:
            Gradient of shape (n_samples,)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        n = len(y_true)
        return -2 * (y_true - y_pred) / n
    
    @staticmethod
    def compute_with_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute both loss and gradient efficiently
        
        Args:
            y_true: True target values of shape (n_samples,)
            y_pred: Predicted values of shape (n_samples,)
            
        Returns:
            Tuple of (loss, gradient)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        n = len(y_true)
        diff = y_true - y_pred
        loss = np.mean(diff ** 2)
        gradient = -2 * diff / n
        
        return loss, gradient
