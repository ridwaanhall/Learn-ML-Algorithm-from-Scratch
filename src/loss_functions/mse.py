"""
Mean Squared Error (MSE) Loss Function

Mathematical Formula:
MSE = (1/n) * Σ(y_true - y_pred)²

Where:
- n = number of samples
- y_true = actual target values
- y_pred = predicted values

Use Cases:
- Regression problems where you want to penalize large errors heavily
- When outliers should have significant impact on the model
- Default choice for most regression tasks

When NOT to use:
- When you have many outliers (use MAE or Huber instead)
- For classification tasks (use Cross Entropy instead)
- When you want equal penalty for all errors regardless of magnitude
"""

import numpy as np


class MSE:
    """Mean Squared Error Loss Function
    
    This is the most common loss function for regression tasks.
    It penalizes larger errors more heavily than smaller ones due to the squaring operation.
    """
    
    def __init__(self):
        """Initialize MSE loss function"""
        self.name = "Mean Squared Error"
        
    def forward(self, y_true, y_pred):
        """
        Calculate the MSE loss
        
        Args:
            y_true (np.ndarray): True target values, shape (n_samples,) or (n_samples, n_features)
            y_pred (np.ndarray): Predicted values, same shape as y_true
            
        Returns:
            float: MSE loss value
            
        Mathematical Steps:
        1. Calculate differences: diff = y_true - y_pred
        2. Square the differences: squared_diff = diff²
        3. Take mean: mse = mean(squared_diff)
        """
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Check shapes match
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        # Calculate MSE: (1/n) * Σ(y_true - y_pred)²
        diff = y_true - y_pred
        squared_diff = diff ** 2
        mse = np.mean(squared_diff)
        
        return mse
    
    def backward(self, y_true, y_pred):
        """
        Calculate the gradient of MSE with respect to predictions
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            np.ndarray: Gradient of MSE with respect to y_pred
            
        Mathematical Derivation:
        MSE = (1/n) * Σ(y_true - y_pred)²
        
        Let: loss = (1/n) * Σ(y_true - y_pred)²
        
        ∂loss/∂y_pred = ∂/∂y_pred [(1/n) * Σ(y_true - y_pred)²]
                      = (1/n) * Σ[2 * (y_true - y_pred) * (-1)]
                      = (-2/n) * Σ(y_true - y_pred)
                      = (2/n) * Σ(y_pred - y_true)
                      = (2/n) * (y_pred - y_true)
        """
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Check shapes match
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        # Calculate gradient: (2/n) * (y_pred - y_true)
        n = y_true.shape[0]  # number of samples
        gradient = (2.0 / n) * (y_pred - y_true)
        
        return gradient
    
    def __call__(self, y_true, y_pred):
        """Allow the class to be called as a function"""
        return self.forward(y_true, y_pred)
    
    def __str__(self):
        return f"MSE Loss Function: L = (1/n) * Σ(y_true - y_pred)²"
    
    def __repr__(self):
        return f"MSE()"


# Example usage and educational demonstration
if __name__ == "__main__":
    # Create MSE loss function
    mse_loss = MSE()
    
    # Example 1: Perfect predictions (loss should be 0)
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    
    loss = mse_loss(y_true, y_pred)
    gradient = mse_loss.backward(y_true, y_pred)
    
    print("Example 1: Perfect Predictions")
    print(f"True values: {y_true}")
    print(f"Predictions: {y_pred}")
    print(f"MSE Loss: {loss}")
    print(f"Gradient: {gradient}")
    print(f"Expected: Loss=0, Gradient=0\n")
    
    # Example 2: Some error
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    
    loss = mse_loss(y_true, y_pred)
    gradient = mse_loss.backward(y_true, y_pred)
    
    print("Example 2: Small Errors")
    print(f"True values: {y_true}")
    print(f"Predictions: {y_pred}")
    print(f"MSE Loss: {loss:.4f}")
    print(f"Gradient: {gradient}")
    print("Note: Gradient points in direction to reduce loss\n")
    
    # Example 3: Large error to show quadratic penalty
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 1.0, 5.0, 2.0])  # Larger errors
    
    loss = mse_loss(y_true, y_pred)
    gradient = mse_loss.backward(y_true, y_pred)
    
    print("Example 3: Larger Errors")
    print(f"True values: {y_true}")
    print(f"Predictions: {y_pred}")
    print(f"MSE Loss: {loss:.4f}")
    print(f"Gradient: {gradient}")
    print("Note: Large errors are penalized heavily due to squaring")
