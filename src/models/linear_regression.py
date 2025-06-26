"""
Linear Regression Model

Mathematical Formula:
ŷ = X * w + b

Where:
- ŷ = predicted values
- X = input features (n_samples × n_features)
- w = weights/coefficients (n_features,)
- b = bias/intercept term

Loss Function (typically MSE):
L = (1/n) * Σ(y_true - ŷ)²

Gradient Update:
∂L/∂w = (2/n) * X^T * (ŷ - y_true)
∂L/∂b = (2/n) * Σ(ŷ - y_true)

Use Cases:
- Predicting continuous values (house prices, temperature, etc.)
- Understanding relationships between variables
- Baseline model for regression problems
- When you need interpretable coefficients
- Simple problems with linear relationships

When NOT to use:
- Non-linear relationships (unless you engineer features)
- Classification problems (use logistic regression)
- When you have more features than samples (use regularization)
- When outliers are a major concern (consider robust regression)

Assumptions:
- Linear relationship between features and target
- Independence of observations
- Homoscedasticity (constant variance of residuals)
- Normal distribution of residuals (for inference)
"""

import numpy as np


class LinearRegression:
    """Linear Regression Model
    
    Implements linear regression using gradient descent optimization.
    Perfect for understanding the fundamentals of machine learning.
    """
    
    def __init__(self):
        """Initialize Linear Regression model"""
        self.weights = None
        self.bias = None
        self.training_history = []
        self.name = "Linear Regression"
        
    def _initialize_parameters(self, n_features):
        """
        Initialize weights and bias
        
        Args:
            n_features (int): Number of input features
        """
        # Initialize weights with small random values
        # Using normal distribution with small standard deviation
        self.weights = np.random.normal(0, 0.01, n_features)
        
        # Initialize bias to zero
        self.bias = 0.0
        
    def _add_bias_term(self, X):
        """
        Add bias term to feature matrix
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Feature matrix with bias column
        """
        # Add column of ones for bias term
        bias_column = np.ones((X.shape[0], 1))
        return np.hstack([bias_column, X])
    
    def predict(self, X):
        """
        Make predictions using the linear model
        
        Args:
            X (np.ndarray): Input features, shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predictions, shape (n_samples,)
            
        Mathematical Steps:
        1. Linear combination: z = X * w + b
        2. Return z (no activation function for regression)
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure X is numpy array
        X = np.array(X)
        
        # Handle 1D input (single sample)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Linear prediction: ŷ = X * w + b
        predictions = np.dot(X, self.weights) + self.bias
        
        return predictions
    
    def fit(self, X, y, loss_function, optimizer, epochs=1000, verbose=True):
        """
        Train the linear regression model
        
        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Training targets, shape (n_samples,)
            loss_function: Loss function object (e.g., MSE)
            optimizer: Optimizer object (e.g., GradientDescent, Adam)
            epochs (int): Number of training epochs
            verbose (bool): Whether to print training progress
            
        Returns:
            dict: Training history with losses
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Handle 1D input
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self._initialize_parameters(n_features)
        
        # Reset training history
        self.training_history = []
        
        # Reset optimizer state
        if hasattr(optimizer, 'reset'):
            optimizer.reset()
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass: make predictions
            y_pred = self.predict(X)
            
            # Calculate loss
            loss = loss_function.forward(y, y_pred)
            
            # Backward pass: calculate gradients
            loss_grad = loss_function.backward(y, y_pred)
            
            # Calculate parameter gradients
            gradients = self._calculate_gradients(X, loss_grad)
            
            # Update parameters using optimizer
            params = {'weights': self.weights, 'bias': self.bias}
            updated_params = optimizer.update(params, gradients)
            
            self.weights = updated_params['weights']
            self.bias = updated_params['bias']
            
            # Store training history
            self.training_history.append({
                'epoch': epoch,
                'loss': loss,
                'weights': self.weights.copy(),
                'bias': self.bias
            })
            
            # Print progress
            if verbose and (epoch % (epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}/{epochs}: Loss = {loss:.6f}")
        
        return {
            'history': self.training_history,
            'final_loss': loss,
            'final_weights': self.weights,
            'final_bias': self.bias
        }
    
    def _calculate_gradients(self, X, loss_grad):
        """
        Calculate gradients of loss with respect to parameters
        
        Args:
            X (np.ndarray): Input features
            loss_grad (np.ndarray): Gradient of loss with respect to predictions
            
        Returns:
            dict: Gradients for weights and bias
        """
        n_samples = X.shape[0]
        
        # Gradient with respect to weights: ∂L/∂w = (1/n) * X^T * ∂L/∂ŷ
        weight_grad = np.dot(X.T, loss_grad) / n_samples
        
        # Gradient with respect to bias: ∂L/∂b = (1/n) * Σ(∂L/∂ŷ)
        bias_grad = np.mean(loss_grad)
        
        return {
            'weights': weight_grad,
            'bias': bias_grad
        }
    
    def get_parameters(self):
        """Get current model parameters"""
        return {
            'weights': self.weights.copy() if self.weights is not None else None,
            'bias': self.bias
        }
    
    def score(self, X, y, metric='r2'):
        """
        Evaluate model performance
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): True targets
            metric (str): Evaluation metric ('r2', 'mse', 'mae')
            
        Returns:
            float: Performance score
        """
        y_pred = self.predict(X)
        
        if metric == 'r2':
            # R-squared score
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
        elif metric == 'mse':
            # Mean Squared Error
            return np.mean((y - y_pred) ** 2)
        elif metric == 'mae':
            # Mean Absolute Error
            return np.mean(np.abs(y - y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def __str__(self):
        if self.weights is not None:
            return f"LinearRegression(weights={self.weights}, bias={self.bias:.4f})"
        else:
            return "LinearRegression(untrained)"
    
    def __repr__(self):
        return "LinearRegression()"


# Example usage and educational demonstration
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    from src.loss_functions.mse import MSE
    from src.optimization.gradient_descent import GradientDescent
    from src.optimization.adam import Adam
    
    print("=== Linear Regression Educational Examples ===\n")
    
    # Example 1: Simple 1D regression
    print("Example 1: Simple 1D Linear Regression")
    print("Fitting y = 2x + 1 with some noise")
    
    # Generate synthetic data: y = 2x + 1 + noise
    np.random.seed(42)
    X_1d = np.random.uniform(-2, 2, 50).reshape(-1, 1)
    y_1d = 2 * X_1d.flatten() + 1 + np.random.normal(0, 0.1, 50)
    
    # Create and train model
    model = LinearRegression()
    loss_fn = MSE()
    optimizer = GradientDescent(learning_rate=0.1)
    
    print(f"Training data: {len(X_1d)} samples")
    print(f"True relationship: y = 2x + 1")
    
    # Train model
    history = model.fit(X_1d, y_1d, loss_fn, optimizer, epochs=1000, verbose=False)
    
    print(f"Learned weights: {model.weights}")
    print(f"Learned bias: {model.bias:.4f}")
    print(f"Final loss: {history['final_loss']:.6f}")
    print(f"R² score: {model.score(X_1d, y_1d, 'r2'):.4f}")
    
    # Example 2: Multi-dimensional regression
    print("\n" + "="*50)
    print("Example 2: Multi-dimensional Linear Regression")
    print("Fitting y = 3x₁ + 2x₂ - x₃ + 5 with noise")
    
    # Generate synthetic data: y = 3x₁ + 2x₂ - x₃ + 5 + noise
    np.random.seed(42)
    X_multi = np.random.uniform(-1, 1, (100, 3))
    true_weights = np.array([3, 2, -1])
    true_bias = 5
    y_multi = np.dot(X_multi, true_weights) + true_bias + np.random.normal(0, 0.1, 100)
    
    # Create and train model with Adam optimizer
    model_multi = LinearRegression()
    adam_optimizer = Adam(learning_rate=0.01)
    
    print(f"Training data: {len(X_multi)} samples, {X_multi.shape[1]} features")
    print(f"True weights: {true_weights}")
    print(f"True bias: {true_bias}")
    
    # Train model
    history_multi = model_multi.fit(X_multi, y_multi, loss_fn, adam_optimizer, 
                                   epochs=1000, verbose=False)
    
    print(f"Learned weights: {model_multi.weights}")
    print(f"Learned bias: {model_multi.bias:.4f}")
    print(f"Final loss: {history_multi['final_loss']:.6f}")
    print(f"R² score: {model_multi.score(X_multi, y_multi, 'r2'):.4f}")
    
    # Example 3: Comparing optimizers
    print("\n" + "="*50)
    print("Example 3: Comparing Different Optimizers")
    
    optimizers = [
        ("Gradient Descent", GradientDescent(learning_rate=0.01)),
        ("Adam", Adam(learning_rate=0.01)),
    ]
    
    print("Training same dataset with different optimizers:")
    
    for opt_name, optimizer in optimizers:
        model_comp = LinearRegression()
        history_comp = model_comp.fit(X_multi, y_multi, loss_fn, optimizer, 
                                     epochs=500, verbose=False)
        
        r2_score = model_comp.score(X_multi, y_multi, 'r2')
        print(f"{opt_name:20s}: Final Loss = {history_comp['final_loss']:.6f}, "
              f"R² = {r2_score:.4f}")
    
    print("\nKey Learning Points:")
    print("1. Linear regression finds the best linear relationship between features and target")
    print("2. The model learns weights and bias to minimize the chosen loss function")
    print("3. Different optimizers can affect training speed and final performance")
    print("4. R² score measures how well the model explains the variance in the data")
    print("5. With sufficient data and proper features, linear regression can be very effective")
