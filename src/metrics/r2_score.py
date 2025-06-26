"""
R-squared (Coefficient of Determination) metric implementation.

This module implements the R² score metric for regression problems.
R² represents the proportion of the variance in the dependent variable
that is predictable from the independent variables.
"""
import numpy as np

class R2Score:
    """
    R-squared (coefficient of determination) metric for regression.
    
    R² = 1 - (SS_res / SS_tot)
    where:
    - SS_res = Σ(y_true - y_pred)²
    - SS_tot = Σ(y_true - y_mean)²
    """
    
    def __init__(self):
        """Initialize R² score metric."""
        pass
    
    def calculate(self, y_true, y_pred):
        """
        Calculate R² score.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            
        Returns:
            float: R² score (can be negative for very poor models)
        """
        # Convert to numpy arrays if needed
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Calculate sum of squares of residuals
        ss_res = np.sum((y_true - y_pred) ** 2)
        
        # Calculate total sum of squares
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        
        # Handle edge case where all y_true values are the same
        if ss_tot == 0:
            if ss_res == 0:
                return 1.0  # Perfect prediction
            else:
                return 0.0  # No variance to explain
        
        # Calculate R²
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def calculate_adjusted(self, y_true, y_pred, n_features):
        """
        Calculate adjusted R² score.
        
        Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - p - 1)]
        where n is the number of samples and p is the number of features.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            n_features (int): Number of features used in the model
            
        Returns:
            float: Adjusted R² score
        """
        r2 = self.calculate(y_true, y_pred)
        n = len(y_true)
        
        if n <= n_features + 1:
            return float('-inf')  # Not enough samples
        
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
        return adjusted_r2
    
    def explained_variance_score(self, y_true, y_pred):
        """
        Calculate explained variance score.
        
        Explained Variance = 1 - Var(y_true - y_pred) / Var(y_true)
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            
        Returns:
            float: Explained variance score
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        var_residual = np.var(y_true - y_pred)
        var_y_true = np.var(y_true)
        
        if var_y_true == 0:
            return 1.0 if var_residual == 0 else 0.0
        
        return 1 - (var_residual / var_y_true)
