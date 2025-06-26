"""
Softmax Activation Function

Mathematical Formula:
softmax(x_i) = e^(x_i) / Σ(e^(x_j)) for j=1 to n

Where:
- x_i = input to neuron i
- n = number of neurons (classes)

Properties:
- Output is a probability distribution
- All outputs sum to 1
- Each output is between 0 and 1
- Differentiable everywhere

Use Cases:
- Multi-class classification output layer
- Converting logits to probabilities
- Attention mechanisms
- When you need a probability distribution

When NOT to use:
- Binary classification (use sigmoid)
- Regression problems
- Hidden layers (use ReLU family)
- When you don't need probabilities

Mathematical Properties:
- Σ softmax(x_i) = 1
- 0 ≤ softmax(x_i) ≤ 1
- Translation invariant: softmax(x + c) = softmax(x)
"""

import numpy as np


class Softmax:
    """Softmax Activation Function
    
    Converts a vector of real numbers into a probability distribution.
    Essential for multi-class classification tasks.
    """
    
    def __init__(self):
        """Initialize Softmax activation function"""
        self.name = "Softmax"
        
    def forward(self, x):
        """
        Apply softmax activation function
        
        Args:
            x (np.ndarray): Input logits, shape (..., n_classes)
            
        Returns:
            np.ndarray: Softmax probabilities, same shape as input
            
        Mathematical Steps:
        1. Subtract max for numerical stability: x_stable = x - max(x)
        2. Compute exponentials: exp_x = e^(x_stable)
        3. Compute sum: sum_exp = Σ(exp_x)
        4. Normalize: softmax = exp_x / sum_exp
        """
        # Ensure input is numpy array
        x = np.array(x)
        
        # Handle both 1D and 2D inputs
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Numerical stability: subtract max from each row
        # This doesn't change the result due to translation invariance
        x_stable = x - np.max(x, axis=1, keepdims=True)
        
        # Compute exponentials
        exp_x = np.exp(x_stable)
        
        # Compute softmax: normalize by sum
        softmax_output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        # Store for backward pass
        self.last_output = softmax_output
        
        # Return original shape if input was 1D
        if squeeze_output:
            softmax_output = softmax_output.squeeze(0)
        
        return softmax_output
    
    def backward(self, grad_output):
        """
        Calculate the gradient of softmax
        
        Args:
            grad_output (np.ndarray): Gradient from the next layer
            
        Returns:
            np.ndarray: Gradient with respect to input
            
        Mathematical Derivation:
        For softmax S_i = e^(x_i) / Σ e^(x_j), the Jacobian is:
        
        ∂S_i/∂x_j = S_i * (δ_ij - S_j)
        
        Where δ_ij is the Kronecker delta (1 if i=j, 0 otherwise)
        
        This means:
        - ∂S_i/∂x_i = S_i * (1 - S_i)     [diagonal elements]
        - ∂S_i/∂x_j = -S_i * S_j          [off-diagonal elements]
        
        For the full gradient using chain rule:
        ∂L/∂x_i = Σ_k (∂L/∂S_k * ∂S_k/∂x_i)
                = ∂L/∂S_i * S_i * (1 - S_i) + Σ_{k≠i} (∂L/∂S_k * (-S_k * S_i))
                = S_i * (∂L/∂S_i - Σ_k (∂L/∂S_k * S_k))
        """
        if not hasattr(self, 'last_output'):
            raise ValueError("Must call forward() before backward()")
        
        # Handle 1D case
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)
            softmax_out = self.last_output.reshape(1, -1)
            squeeze_output = True
        else:
            softmax_out = self.last_output
            squeeze_output = False
        
        # Softmax gradient: S_i * (∂L/∂S_i - Σ_k(∂L/∂S_k * S_k))
        # This is equivalent to: softmax * (grad_output - (grad_output * softmax).sum(axis=1, keepdims=True))
        sum_term = np.sum(grad_output * softmax_out, axis=1, keepdims=True)
        grad_input = softmax_out * (grad_output - sum_term)
        
        # Return original shape if input was 1D
        if squeeze_output:
            grad_input = grad_input.squeeze(0)
        
        return grad_input
    
    def __call__(self, x):
        """Allow the class to be called as a function"""
        return self.forward(x)
    
    def __str__(self):
        return "Softmax: S_i = e^(x_i) / Σe^(x_j)"
    
    def __repr__(self):
        return "Softmax()"


# Example usage and educational demonstration
if __name__ == "__main__":
    # Create softmax activation function
    softmax = Softmax()
    
    print("=== Softmax Activation Function Educational Examples ===\n")
    
    # Example 1: Basic behavior
    print("Example 1: Basic Softmax Behavior")
    
    # Test with logits for 3 classes
    logits = np.array([2.0, 1.0, 0.1])
    probabilities = softmax(logits)
    
    print(f"Input logits: {logits}")
    print(f"Softmax output: {probabilities}")
    print(f"Sum of probabilities: {np.sum(probabilities):.6f}")
    print(f"Predicted class: {np.argmax(probabilities)}")
    
    print("\nKey observations:")
    print("- Largest logit gets highest probability")
    print("- All probabilities sum to 1.0")
    print("- All probabilities are positive")
    
    # Example 2: Temperature effect (sharpness)
    print("\n" + "="*50)
    print("Example 2: Effect of Temperature (Scaling)")
    
    base_logits = np.array([2.0, 1.0, 0.1])
    temperatures = [0.5, 1.0, 2.0, 10.0]
    
    print("Temperature | Logits after scaling | Probabilities")
    print("-" * 60)
    
    for temp in temperatures:
        scaled_logits = base_logits / temp
        probs = softmax(scaled_logits)
        
        print(f"T = {temp:4.1f}   | {scaled_logits} | {probs}")
    
    print("\nTemperature insights:")
    print("- T < 1: Sharper distribution (more confident)")
    print("- T = 1: Normal softmax")
    print("- T > 1: Softer distribution (less confident)")
    print("- T → ∞: Uniform distribution")
    
    # Example 3: Multi-sample batch
    print("\n" + "="*50)
    print("Example 3: Batch Processing")
    
    # Multiple samples, each with 4 class logits
    batch_logits = np.array([
        [3.0, 1.0, 0.2, 0.1],  # Sample 1: clearly class 0
        [1.0, 2.5, 1.0, 0.8],  # Sample 2: likely class 1
        [0.1, 0.2, 0.1, 2.0],  # Sample 3: clearly class 3
        [1.0, 1.0, 1.0, 1.0]   # Sample 4: uniform (uncertain)
    ])
    
    batch_probs = softmax(batch_logits)
    predictions = np.argmax(batch_probs, axis=1)
    confidence = np.max(batch_probs, axis=1)
    
    print("Sample | Logits                | Probabilities              | Pred | Conf")
    print("-" * 85)
    
    for i in range(len(batch_logits)):
        logits_str = f"[{', '.join(f'{x:4.1f}' for x in batch_logits[i])}]"
        probs_str = f"[{', '.join(f'{x:.3f}' for x in batch_probs[i])}]"
        print(f"   {i+1}   | {logits_str} | {probs_str} | {predictions[i]:4d} | {confidence[i]:.3f}")
    
    # Example 4: Numerical stability demonstration
    print("\n" + "="*50)
    print("Example 4: Numerical Stability")
    
    # Large logits that could cause overflow
    large_logits = np.array([1000, 999, 998])
    
    print("Large logits:", large_logits)
    
    # Naive implementation (would overflow)
    print("Naive exp:", f"Would compute e^{large_logits[0]} = overflow!")
    
    # Our stable implementation
    stable_probs = softmax(large_logits)
    print("Stable softmax:", stable_probs)
    print("Sum:", np.sum(stable_probs))
    
    # Show the stability trick
    print(f"\nStability trick: subtract max({np.max(large_logits)})")
    stable_logits = large_logits - np.max(large_logits)
    print("After subtraction:", stable_logits)
    print("These are manageable for exp()")
    
    # Example 5: Gradient behavior
    print("\n" + "="*50)
    print("Example 5: Gradient Analysis")
    
    # Test gradient with different predictions
    test_logits = np.array([2.0, 1.0, 0.5])
    test_probs = softmax(test_logits)
    
    # Simulate different gradient inputs
    grad_scenarios = [
        ("Uniform gradient", np.array([1.0, 1.0, 1.0])),
        ("Focus on class 0", np.array([1.0, 0.0, 0.0])),
        ("Opposite to prediction", np.array([0.0, 0.0, 1.0])),
    ]
    
    print("Scenario              | Input Grad | Softmax Grad | Notes")
    print("-" * 70)
    
    for name, grad_in in grad_scenarios:
        grad_out = softmax.backward(grad_in)
        
        if name == "Uniform gradient":
            note = "Sums to zero (probability constraint)"
        elif name == "Focus on class 0":
            note = f"Pushes up class 0 (prob={test_probs[0]:.3f})"
        else:
            note = f"Pushes up class 2 (prob={test_probs[2]:.3f})"
        
        grad_in_str = f"[{', '.join(f'{x:.1f}' for x in grad_in)}]"
        grad_out_str = f"[{', '.join(f'{x:+.3f}' for x in grad_out)}]"
        
        print(f"{name:20s} | {grad_in_str:10s} | {grad_out_str:15s} | {note}")
    
    print(f"\nGradient sum: {np.sum(softmax.backward(np.array([1.0, 1.0, 1.0]))):.6f}")
    print("Note: Gradients always sum to zero due to probability constraint")
    
    # Example 6: Cross-entropy combination
    print("\n" + "="*50)
    print("Example 6: Softmax + Cross-Entropy (Common Combination)")
    
    # True labels (one-hot)
    y_true = np.array([1, 0, 0])  # Class 0
    
    # Predictions
    logits = np.array([2.0, 1.0, 0.5])
    probs = softmax(logits)
    
    # Cross-entropy loss
    cross_entropy = -np.sum(y_true * np.log(probs + 1e-15))
    
    print(f"True label: {y_true} (class {np.argmax(y_true)})")
    print(f"Logits: {logits}")
    print(f"Softmax probs: {probs}")
    print(f"Cross-entropy loss: {cross_entropy:.4f}")
    
    # Gradient of softmax + cross-entropy simplifies!
    simple_grad = probs - y_true
    print(f"Combined gradient: {simple_grad}")
    print("Note: Softmax + Cross-entropy gradient = predictions - true_labels")
    
    print("\n" + "="*50)
    print("Summary: When to Use Softmax")
    print("✅ Multi-class classification output layer")
    print("✅ Converting logits to probabilities")
    print("✅ Attention mechanisms")
    print("✅ When you need a probability distribution")
    print("❌ Binary classification (use sigmoid)")
    print("❌ Regression problems")
    print("❌ Hidden layers (use ReLU)")
    print("🔧 Always use with cross-entropy loss for classification")
