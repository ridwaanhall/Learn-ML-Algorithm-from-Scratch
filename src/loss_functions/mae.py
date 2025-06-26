"""
Mean Absolute Error (MAE) Loss Function

Mathematical Formula:
MAE = (1/n) * Σ|y_true - y_pred|

Where:
- n = number of samples
- y_true = actual target values
- y_pred = predicted values

Use Cases:
- Regression problems with outliers
- When all errors should be weighted equally
- Robust regression (less sensitive to outliers than MSE)
- When you want linear penalty for errors

When NOT to use:
- When large errors should be penalized more (use MSE)
- For classification tasks (use Cross Entropy)
- When you need everywhere-differentiable loss (MAE not differentiable at 0)

Characteristics:
- Robust to outliers (unlike MSE)
- All errors weighted equally
- Not differentiable at zero
- More interpretable (same units as target)
"""

import numpy as np


class MAE:
    """Mean Absolute Error Loss Function
    
    Less sensitive to outliers compared to MSE. Provides equal weight
    to all errors regardless of magnitude.
    """
    
    def __init__(self):
        """Initialize MAE loss function"""
        self.name = "Mean Absolute Error"
        
    def forward(self, y_true, y_pred):
        """
        Calculate the MAE loss
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            float: MAE loss value
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        # Calculate MAE: (1/n) * Σ|y_true - y_pred|
        absolute_errors = np.abs(y_true - y_pred)
        mae = np.mean(absolute_errors)
        
        return mae
    
    def backward(self, y_true, y_pred):
        """
        Calculate the gradient of MAE with respect to predictions
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            np.ndarray: Gradient of MAE with respect to y_pred
            
        Mathematical Derivation:
        MAE = (1/n) * Σ|y_true - y_pred|
        
        ∂MAE/∂y_pred = (1/n) * sign(y_pred - y_true)
        
        Where sign(x) = 1 if x > 0, -1 if x < 0, 0 if x = 0
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        n = y_true.shape[0]
        
        # Calculate gradient: (1/n) * sign(y_pred - y_true)
        diff = y_pred - y_true
        gradient = np.sign(diff) / n
        
        return gradient
    
    def __call__(self, y_true, y_pred):
        """Allow the class to be called as a function"""
        return self.forward(y_true, y_pred)
    
    def __str__(self):
        return "MAE Loss Function: L = (1/n) * Σ|y_true - y_pred|"
    
    def __repr__(self):
        return "MAE()"


# Example usage and educational demonstration
if __name__ == "__main__":
    # Create MAE loss function
    mae_loss = MAE()
    
    print("=== MAE Loss Function Educational Examples ===\n")
    
    # Example 1: Compare MAE vs MSE with outliers
    print("Example 1: MAE vs MSE with Outliers")
    
    # Dataset with outlier
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_good = np.array([1.1, 1.9, 3.1, 3.9, 5.1])  # Small errors
    y_pred_outlier = np.array([1.1, 1.9, 3.1, 3.9, 10.0])  # One large error
    
    # Calculate MAE for both cases
    mae_good = mae_loss(y_true, y_pred_good)
    mae_outlier = mae_loss(y_true, y_pred_outlier)
    
    # For comparison, let's calculate MSE as well
    mse_good = np.mean((y_true - y_pred_good) ** 2)
    mse_outlier = np.mean((y_true - y_pred_outlier) ** 2)
    
    print("Good predictions (small errors):")
    print(f"  True:  {y_true}")
    print(f"  Pred:  {y_pred_good}")
    print(f"  MAE:   {mae_good:.4f}")
    print(f"  MSE:   {mse_good:.4f}")
    
    print("\nWith outlier (one large error):")
    print(f"  True:  {y_true}")
    print(f"  Pred:  {y_pred_outlier}")
    print(f"  MAE:   {mae_outlier:.4f}")
    print(f"  MSE:   {mse_outlier:.4f}")
    
    print(f"\nMAE increase: {mae_outlier/mae_good:.2f}x")
    print(f"MSE increase: {mse_outlier/mse_good:.2f}x")
    print("Note: MSE is much more sensitive to outliers!")
    
    # Example 2: Gradient behavior
    print("\n" + "="*50)
    print("Example 2: MAE Gradient Behavior")
    
    # Different error magnitudes
    y_true = np.array([0.0, 0.0, 0.0])
    predictions = [
        np.array([-2.0, -0.1, 0.0]),  # Negative errors and zero error
        np.array([0.1, 1.0, 5.0])     # Positive errors
    ]
    
    for i, y_pred in enumerate(predictions, 1):
        mae = mae_loss(y_true, y_pred)
        gradient = mae_loss.backward(y_true, y_pred)
        
        print(f"\nCase {i}:")
        print(f"  Predictions: {y_pred}")
        print(f"  Errors:      {y_pred - y_true}")
        print(f"  MAE:         {mae:.4f}")
        print(f"  Gradients:   {gradient}")
        print(f"  Note: Gradient is just the sign of the error!")
    
    # Example 3: Comparing different loss functions
    print("\n" + "="*50)
    print("Example 3: Loss Function Comparison")
    
    # Create scenarios with different error patterns
    scenarios = [
        ("Small uniform errors", np.array([0.1, 0.1, 0.1, 0.1])),
        ("One large error", np.array([0.0, 0.0, 0.0, 1.0])),
        ("Mixed errors", np.array([0.1, 0.5, 0.2, 0.8])),
    ]
    
    y_true_base = np.array([1.0, 2.0, 3.0, 4.0])
    
    print("Scenario               | MAE    | MSE    | MSE/MAE Ratio")
    print("-" * 60)
    
    for name, errors in scenarios:
        y_pred = y_true_base + errors
        
        mae = mae_loss(y_true_base, y_pred)
        mse = np.mean(errors ** 2)
        ratio = mse / mae if mae > 0 else 0
        
        print(f"{name:22s} | {mae:.4f} | {mse:.4f} | {ratio:.2f}")
    
    print("\nInsights:")
    print("- MAE gives equal weight to all errors")
    print("- MSE penalizes large errors more heavily")
    print("- Higher MSE/MAE ratio indicates presence of outliers")
    print("- MAE gradient is constant (±1/n), MSE gradient grows with error")
