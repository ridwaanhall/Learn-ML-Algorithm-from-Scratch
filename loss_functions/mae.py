"""
Mean Absolute Error (MAE) loss function
"""
import numpy as np
from typing import Tuple


class MAELoss:
    """
    Mean Absolute Error loss function for regression tasks
    
    MAE = (1/n) * Σ|y_true - y_pred|
    """
    
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute MAE loss
        
        Args:
            y_true: True target values of shape (n_samples,)
            y_pred: Predicted values of shape (n_samples,)
            
        Returns:
            MAE loss value
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradient of MAE loss with respect to predictions
        
        Args:
            y_true: True target values of shape (n_samples,)
            y_pred: Predicted values of shape (n_samples,)
            
        Returns:
            Gradient of shape (n_samples,)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        n = len(y_true)
        diff = y_true - y_pred
        # Gradient is -sign(diff) / n
        gradient = -np.sign(diff) / n
        
        # Handle the case where diff = 0 (gradient is undefined)
        gradient[diff == 0] = 0
        
        return gradient
    
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
        loss = np.mean(np.abs(diff))
        
        # Gradient computation
        gradient = -np.sign(diff) / n
        gradient[diff == 0] = 0
        
        return loss, gradient
