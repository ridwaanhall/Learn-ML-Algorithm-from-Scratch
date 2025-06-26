"""
Huber Loss Function

Mathematical Formula:
For |y_true - y_pred| ≤ δ:
    L = 0.5 * (y_true - y_pred)²

For |y_true - y_pred| > δ:
    L = δ * |y_true - y_pred| - 0.5 * δ²

Where δ (delta) is a threshold parameter (typically 1.0)

Use Cases:
- Robust regression (less sensitive to outliers than MSE)
- When you want quadratic loss for small errors, linear for large errors
- Computer vision applications (object detection)
- When you have some outliers but want smooth gradients

When NOT to use:
- When you want all errors treated equally (use MAE)
- When outliers should be heavily penalized (use MSE)
- Classification problems (use Cross Entropy)

Characteristics:
- Combines benefits of MSE and MAE
- Quadratic for small errors (smooth gradients)
- Linear for large errors (robust to outliers)
- Parameter δ controls the transition point
"""

import numpy as np


class HuberLoss:
    """Huber Loss Function
    
    A robust loss function that is quadratic for small errors
    and linear for large errors. Best of both MSE and MAE.
    """
    
    def __init__(self, delta=1.0):
        """
        Initialize Huber loss function
        
        Args:
            delta (float): Threshold for switching from quadratic to linear loss
        """
        self.delta = delta
        self.name = f"Huber Loss (δ={delta})"
        
    def forward(self, y_true, y_pred):
        """
        Calculate the Huber loss
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            float: Huber loss value
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        # Calculate absolute error
        abs_error = np.abs(y_true - y_pred)
        
        # Huber loss: quadratic for small errors, linear for large errors
        is_small_error = abs_error <= self.delta
        
        # For small errors: 0.5 * error²
        quadratic_loss = 0.5 * (y_true - y_pred) ** 2
        
        # For large errors: δ * |error| - 0.5 * δ²
        linear_loss = self.delta * abs_error - 0.5 * self.delta ** 2
        
        # Combine based on error magnitude
        loss = np.where(is_small_error, quadratic_loss, linear_loss)
        
        return np.mean(loss)
    
    def backward(self, y_true, y_pred):
        """
        Calculate the gradient of Huber loss
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            np.ndarray: Gradient with respect to y_pred
            
        Mathematical Derivation:
        For |error| ≤ δ: ∂L/∂y_pred = (y_pred - y_true) / n
        For |error| > δ:  ∂L/∂y_pred = δ * sign(y_pred - y_true) / n
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        n = y_true.shape[0]
        error = y_pred - y_true
        abs_error = np.abs(error)
        
        # For small errors: gradient like MSE
        # For large errors: gradient like MAE
        is_small_error = abs_error <= self.delta
        
        gradient = np.where(
            is_small_error,
            error,                              # MSE-like gradient
            self.delta * np.sign(error)         # MAE-like gradient
        )
        
        return gradient / n
    
    def __call__(self, y_true, y_pred):
        """Allow the class to be called as a function"""
        return self.forward(y_true, y_pred)
    
    def __str__(self):
        return f"Huber Loss (δ={self.delta}): Quadratic for |error| ≤ δ, Linear for |error| > δ"
    
    def __repr__(self):
        return f"HuberLoss(delta={self.delta})"


# Example usage and educational demonstration
if __name__ == "__main__":
    print("=== Huber Loss Function Educational Examples ===\n")
    
    # Example 1: Compare different delta values
    print("Example 1: Effect of Delta Parameter")
    
    deltas = [0.5, 1.0, 2.0]
    huber_losses = [HuberLoss(delta=d) for d in deltas]
    
    # Test with different error magnitudes
    y_true = np.array([0.0, 0.0, 0.0])
    test_errors = [0.2, 1.0, 3.0]  # Small, medium, large error
    
    print("Error | δ=0.5  | δ=1.0  | δ=2.0  | Notes")
    print("-" * 55)
    
    for error in test_errors:
        y_pred = np.array([error, error, error])
        
        losses = [hl(y_true, y_pred) for hl in huber_losses]
        
        if error <= 0.5:
            note = "All quadratic"
        elif error <= 1.0:
            note = "δ=0.5 linear"
        elif error <= 2.0:
            note = "δ≤1.0 linear"
        else:
            note = "All linear"
        
        print(f"{error:5.1f} | {losses[0]:6.3f} | {losses[1]:6.3f} | {losses[2]:6.3f} | {note}")
    
    # Example 2: Comparison with MSE and MAE
    print("\n" + "="*50)
    print("Example 2: Huber vs MSE vs MAE")
    
    # Create test scenario with outliers
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    scenarios = [
        ("Good predictions", np.array([1.1, 1.9, 3.1, 3.9, 5.1])),
        ("One outlier", np.array([1.1, 1.9, 3.1, 3.9, 8.0])),
        ("Large outlier", np.array([1.1, 1.9, 3.1, 3.9, 15.0])),
    ]
    
    huber = HuberLoss(delta=1.0)
    
    print("Scenario          | Huber  | MSE    | MAE    | Huber Properties")
    print("-" * 75)
    
    for name, y_pred in scenarios:
        huber_loss = huber(y_true, y_pred)
        mse_loss = np.mean((y_true - y_pred) ** 2)
        mae_loss = np.mean(np.abs(y_true - y_pred))
        
        # Analyze which errors are in quadratic vs linear regime
        errors = np.abs(y_true - y_pred)
        quad_count = np.sum(errors <= huber.delta)
        linear_count = np.sum(errors > huber.delta)
        
        properties = f"{quad_count} quad, {linear_count} linear"
        
        print(f"{name:17s} | {huber_loss:6.3f} | {mse_loss:6.3f} | {mae_loss:6.3f} | {properties}")
    
    # Example 3: Gradient behavior
    print("\n" + "="*50)
    print("Example 3: Gradient Behavior Analysis")
    
    huber = HuberLoss(delta=1.0)
    
    # Test gradients at different error levels
    test_errors = np.array([-3, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 3])
    y_true = np.zeros_like(test_errors)
    y_pred = test_errors  # So error = y_pred - y_true = test_errors
    
    print("Error | Huber Loss | Huber Grad | MSE Grad | MAE Grad | Regime")
    print("-" * 70)
    
    for i, error in enumerate(test_errors):
        y_t = np.array([0.0])
        y_p = np.array([error])
        
        # Calculate losses and gradients
        huber_loss = huber(y_t, y_p)
        huber_grad = huber.backward(y_t, y_p)[0]
        
        mse_grad = 2 * error  # MSE gradient
        mae_grad = np.sign(error)  # MAE gradient
        
        regime = "Quadratic" if abs(error) <= huber.delta else "Linear"
        
        print(f"{error:5.1f} | {huber_loss:10.4f} | {huber_grad:10.3f} | {mse_grad:8.3f} | {mae_grad:8.1f} | {regime}")
    
    # Example 4: Practical application
    print("\n" + "="*50)
    print("Example 4: Practical Application - Robust Regression")
    
    # Simulate regression data with outliers
    np.random.seed(42)
    
    # Generate clean data: y = 2x + 1
    x_clean = np.linspace(0, 5, 20)
    y_clean = 2 * x_clean + 1
    
    # Add some outliers
    y_with_outliers = y_clean.copy()
    y_with_outliers[5] += 10   # Large positive outlier
    y_with_outliers[15] -= 8   # Large negative outlier
    
    # Compare loss functions
    huber = HuberLoss(delta=1.0)
    
    huber_loss = huber(y_clean, y_with_outliers)
    mse_loss = np.mean((y_clean - y_with_outliers) ** 2)
    mae_loss = np.mean(np.abs(y_clean - y_with_outliers))
    
    print("Regression with outliers:")
    print(f"Data points: {len(y_clean)}")
    print(f"Outliers: 2 points with large deviations")
    print(f"Huber Loss (δ=1.0): {huber_loss:.4f}")
    print(f"MSE Loss:           {mse_loss:.4f}")
    print(f"MAE Loss:           {mae_loss:.4f}")
    
    print("\nInterpretation:")
    print("- MSE heavily penalized by outliers (squared errors)")
    print("- MAE treats all errors equally")
    print("- Huber provides balanced approach: smooth gradients + outlier robustness")
    
    print("\n" + "="*50)
    print("Summary: When to use Huber Loss")
    print("✅ Regression with some outliers")
    print("✅ When you want smooth gradients (better than MAE)")
    print("✅ When you want outlier robustness (better than MSE)")
    print("✅ Computer vision tasks (object detection)")
    print("❌ When all errors should be weighted equally (use MAE)")
    print("❌ When outliers should be heavily penalized (use MSE)")
    print("❌ Classification tasks (use Cross Entropy)")
    print(f"⚙️  Tune δ parameter: smaller δ → more like MAE, larger δ → more like MSE")
