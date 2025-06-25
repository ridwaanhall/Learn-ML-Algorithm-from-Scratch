"""
Regression metrics for model evaluation
"""
import numpy as np
from typing import Optional


class RegressionMetrics:
    """
    Collection of regression evaluation metrics
    """
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error (MSE)
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            MSE value
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error (RMSE)
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        mse = RegressionMetrics.mean_squared_error(y_true, y_pred)
        return np.sqrt(mse)
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error (MAE)
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R² (coefficient of determination) score
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            R² score
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        ss_res = np.sum((y_true - y_pred) ** 2)  # Sum of squares of residuals
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
        """
        Calculate Adjusted R² score
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            n_features: Number of features used in the model
            
        Returns:
            Adjusted R² score
        """
        r2 = RegressionMetrics.r2_score(y_true, y_pred)
        n = len(y_true)
        
        if n <= n_features + 1:
            return float('-inf')  # Undefined when n <= p + 1
        
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return adjusted_r2
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE)
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            epsilon: Small value to avoid division by zero
            
        Returns:
            MAPE value as percentage
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        # Avoid division by zero
        y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
        
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
        return mape
    
    @staticmethod
    def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Median Absolute Error
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Median absolute error
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        return np.median(np.abs(y_true - y_pred))
    
    @staticmethod
    def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Explained Variance Score
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Explained variance score
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        var_y = np.var(y_true)
        var_residual = np.var(y_true - y_pred)
        
        if var_y == 0:
            return 1.0 if var_residual == 0 else 0.0
        
        return 1 - var_residual / var_y
    
    @staticmethod
    def regression_report(y_true: np.ndarray, y_pred: np.ndarray, n_features: Optional[int] = None) -> dict:
        """
        Generate a comprehensive regression report
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            n_features: Number of features (for adjusted R²)
            
        Returns:
            Dictionary containing various regression metrics
        """
        report = {
            'mse': RegressionMetrics.mean_squared_error(y_true, y_pred),
            'rmse': RegressionMetrics.root_mean_squared_error(y_true, y_pred),
            'mae': RegressionMetrics.mean_absolute_error(y_true, y_pred),
            'r2': RegressionMetrics.r2_score(y_true, y_pred),
            'mape': RegressionMetrics.mean_absolute_percentage_error(y_true, y_pred),
            'median_ae': RegressionMetrics.median_absolute_error(y_true, y_pred),
            'explained_variance': RegressionMetrics.explained_variance_score(y_true, y_pred)
        }
        
        if n_features is not None:
            report['adjusted_r2'] = RegressionMetrics.adjusted_r2_score(y_true, y_pred, n_features)
        
        return report
