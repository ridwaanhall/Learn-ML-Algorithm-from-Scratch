"""
Linear Regression implementation from scratch
"""
import numpy as np
from .base_model import BaseRegressor
from ..utils.matrix import MatrixUtils
from ..loss_functions.mse import MSELoss
from ..optimization.sgd import SGDOptimizer
from typing import Optional, Union


class LinearRegression(BaseRegressor):
    """
    Linear Regression using Ordinary Least Squares
    
    Linear model that assumes a linear relationship between input features
    and target variable: y = X * w + b
    """
    
    def __init__(self, fit_intercept: bool = True, solver: str = 'normal_equation', 
                 learning_rate: float = 0.01, max_iterations: int = 1000, 
                 tolerance: float = 1e-6):
        """
        Initialize Linear Regression
        
        Args:
            fit_intercept: Whether to calculate intercept
            solver: Solver to use ('normal_equation', 'sgd')
            learning_rate: Learning rate for SGD solver
            max_iterations: Maximum iterations for SGD solver
            tolerance: Tolerance for convergence
        """
        super().__init__()
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Model parameters
        self.weights = None
        self.intercept = None
        self.loss_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit linear regression model
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        self._validate_input(X, y)
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Add bias column if fitting intercept
        if self.fit_intercept:
            X_with_bias = MatrixUtils.add_bias_column(X)
        else:
            X_with_bias = X.copy()
        
        if self.solver == 'normal_equation':
            self._fit_normal_equation(X_with_bias, y)
        elif self.solver == 'sgd':
            self._fit_sgd(X_with_bias, y)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        self.is_fitted = True
        return self
    
    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit using normal equation: w = (X^T * X)^(-1) * X^T * y
        
        Args:
            X: Feature matrix (with bias if needed)
            y: Target vector
        """
        try:
            # Normal equation
            XtX = MatrixUtils.dot_product(MatrixUtils.transpose(X), X)
            Xty = MatrixUtils.dot_product(MatrixUtils.transpose(X), y)
            weights = np.linalg.solve(XtX, Xty)
            
            if self.fit_intercept:
                self.intercept = weights[0]
                self.weights = weights[1:]
            else:
                self.intercept = 0
                self.weights = weights
                
        except np.linalg.LinAlgError:
            # If matrix is singular, add regularization
            XtX = MatrixUtils.dot_product(MatrixUtils.transpose(X), X)
            XtX += 1e-8 * np.eye(XtX.shape[0])  # Ridge regularization
            Xty = MatrixUtils.dot_product(MatrixUtils.transpose(X), y)
            weights = np.linalg.solve(XtX, Xty)
            
            if self.fit_intercept:
                self.intercept = weights[0]
                self.weights = weights[1:]
            else:
                self.intercept = 0
                self.weights = weights
    
    def _fit_sgd(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit using Stochastic Gradient Descent
        
        Args:
            X: Feature matrix (with bias if needed)
            y: Target vector
        """
        n_features = X.shape[1]
        
        # Initialize weights
        weights = np.random.normal(0, 0.01, n_features)
        
        optimizer = SGDOptimizer(learning_rate=self.learning_rate)
        self.loss_history = []
        
        prev_loss = float('inf')
        
        for iteration in range(self.max_iterations):
            # Forward pass
            y_pred = MatrixUtils.dot_product(X, weights)
            
            # Compute loss
            loss = MSELoss.compute(y, y_pred)
            self.loss_history.append(loss)
            
            # Check convergence
            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss
            
            # Compute gradients
            residuals = y_pred - y
            gradients = MatrixUtils.dot_product(MatrixUtils.transpose(X), residuals) / len(y)
            
            # Update weights
            weights = optimizer.update(weights, gradients)
        
        if self.fit_intercept:
            self.intercept = weights[0]
            self.weights = weights[1:]
        else:
            self.intercept = 0
            self.weights = weights
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted model
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        self._check_fitted()
        self._validate_input(X)
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features, but model was fitted with {self.n_features}")
        
        predictions = MatrixUtils.dot_product(X, self.weights)
        if self.fit_intercept:
            predictions += self.intercept
        
        return predictions
    
    def get_params(self) -> dict:
        """
        Get model parameters
        
        Returns:
            Dictionary of model parameters
        """
        return {
            'weights': self.weights,
            'intercept': self.intercept,
            'fit_intercept': self.fit_intercept,
            'solver': self.solver,
            'learning_rate': self.learning_rate,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance
        }


class RidgeRegression(BaseRegressor):
    """
    Ridge Regression (L2 regularization)
    
    Linear regression with L2 penalty: minimize ||y - Xw||² + α||w||²
    """
    
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True, 
                 solver: str = 'normal_equation', max_iterations: int = 1000, 
                 tolerance: float = 1e-6):
        """
        Initialize Ridge Regression
        
        Args:
            alpha: Regularization strength
            fit_intercept: Whether to calculate intercept
            solver: Solver to use ('normal_equation', 'sgd')
            max_iterations: Maximum iterations for iterative solvers
            tolerance: Tolerance for convergence
        """
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Model parameters
        self.weights = None
        self.intercept = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegression':
        """
        Fit ridge regression model
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        self._validate_input(X, y)
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Center data if fitting intercept
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X.copy()
            y_centered = y.copy()
            X_mean = np.zeros(n_features)
            y_mean = 0
        
        # Solve: (X^T * X + α * I) * w = X^T * y
        XtX = MatrixUtils.dot_product(MatrixUtils.transpose(X_centered), X_centered)
        XtX += self.alpha * np.eye(n_features)
        Xty = MatrixUtils.dot_product(MatrixUtils.transpose(X_centered), y_centered)
        
        self.weights = np.linalg.solve(XtX, Xty)
        
        if self.fit_intercept:
            self.intercept = y_mean - MatrixUtils.dot_product(X_mean, self.weights)
        else:
            self.intercept = 0
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted model
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        self._check_fitted()
        self._validate_input(X)
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features, but model was fitted with {self.n_features}")
        
        predictions = MatrixUtils.dot_product(X, self.weights)
        if self.fit_intercept:
            predictions += self.intercept
        
        return predictions
