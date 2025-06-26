"""
Tanh (Hyperbolic Tangent) Activation Function

Mathematical Formula:
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        = 2*sigmoid(2x) - 1

Derivative:
tanh'(x) = 1 - tanh²(x)

Characteristics:
- Range: (-1, 1)
- Zero-centered output
- S-shaped curve like sigmoid
- Saturates at both ends
- Stronger gradients than sigmoid

Use Cases:
- Hidden layers in small networks
- When you need zero-centered outputs
- RNN/LSTM internal gates
- When sigmoid vanishing gradient is problematic

When NOT to use:
- Deep networks (still has vanishing gradient)
- When you need non-negative outputs
- Most modern deep learning (ReLU family preferred)

Pros:
- Zero-centered (better than sigmoid)
- Stronger gradients than sigmoid
- Symmetric around origin

Cons:
- Still suffers from vanishing gradient
- Computationally expensive
- Saturates for large inputs
"""

import numpy as np


class Tanh:
    """Tanh Activation Function
    
    Zero-centered alternative to sigmoid. Better than sigmoid but
    still not ideal for deep networks due to vanishing gradients.
    """
    
    def __init__(self):
        """Initialize Tanh activation function"""
        self.name = "Tanh"
        
    def forward(self, x):
        """
        Apply tanh activation function
        
        Args:
            x (np.ndarray): Input values of any shape
            
        Returns:
            np.ndarray: Tanh activated values, same shape as input
            
        Mathematical Steps:
        1. For each element x_i: tanh(x_i) = (e^x_i - e^(-x_i)) / (e^x_i + e^(-x_i))
        2. Uses stable computation to avoid overflow
        """
        # Ensure input is numpy array
        x = np.array(x)
        
        # Use numpy's built-in tanh for numerical stability
        output = np.tanh(x)
        
        # Store output for backward pass
        self.last_output = output
        
        return output
    
    def backward(self, grad_output):
        """
        Calculate the gradient of tanh
        
        Args:
            grad_output (np.ndarray): Gradient from the next layer
            
        Returns:
            np.ndarray: Gradient with respect to input
            
        Mathematical Derivation:
        tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        
        Let u = e^x and v = e^(-x), then tanh(x) = (u - v) / (u + v)
        
        Using quotient rule and chain rule:
        d/dx tanh(x) = [(u + v)(u + v) - (u - v)(u - v)] / (u + v)²
                     = [u² + 2uv + v² - u² + 2uv - v²] / (u + v)²
                     = 4uv / (u + v)²
                     = 4e^x * e^(-x) / (e^x + e^(-x))²
                     = 4 / (e^x + e^(-x))²
                     = 1 - tanh²(x)
        
        This elegant form uses the output of the forward pass!
        """
        if not hasattr(self, 'last_output'):
            raise ValueError("Must call forward() before backward()")
        
        # Tanh derivative: tanh'(x) = 1 - tanh²(x)
        tanh_derivative = 1 - self.last_output ** 2
        
        # Apply chain rule
        grad_input = grad_output * tanh_derivative
        
        return grad_input
    
    def __call__(self, x):
        """Allow the class to be called as a function"""
        return self.forward(x)
    
    def __str__(self):
        return "Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))"
    
    def __repr__(self):
        return "Tanh()"


# Example usage and educational demonstration
if __name__ == "__main__":
    # Create tanh activation function
    tanh = Tanh()
    
    print("=== Tanh Activation Function Educational Examples ===\n")
    
    # Example 1: Basic behavior
    print("Example 1: Basic Tanh Behavior")
    
    # Test with various inputs
    test_inputs = np.array([-3, -1, -0.5, 0, 0.5, 1, 3])
    
    print("Input  | Output | Interpretation")
    print("-" * 40)
    
    for x in test_inputs:
        output = tanh(x)
        
        if output < -0.5:
            interp = "Strong negative"
        elif output < -0.1:
            interp = "Weak negative"
        elif output < 0.1:
            interp = "Near zero"
        elif output < 0.5:
            interp = "Weak positive"
        else:
            interp = "Strong positive"
        
        print(f"{x:6.1f} | {output:6.3f} | {interp}")
    
    print("\nKey observations:")
    print("- Large negative inputs → close to -1")
    print("- Zero input → exactly 0 (zero-centered!)")
    print("- Large positive inputs → close to +1")
    print("- Symmetric around origin: tanh(-x) = -tanh(x)")
    
    # Example 2: Compare with Sigmoid
    print("\n" + "="*50)
    print("Example 2: Tanh vs Sigmoid Comparison")
    
    # Import sigmoid for comparison
    from .sigmoid import Sigmoid
    sigmoid = Sigmoid()
    
    test_points = np.array([-2, -1, 0, 1, 2])
    
    print("Input | Tanh   | Sigmoid | Notes")
    print("-" * 45)
    
    for x in test_points:
        tanh_out = tanh(x)
        sig_out = sigmoid(x)
        
        if x == 0:
            note = "Zero-centered vs 0.5"
        elif x > 0:
            note = "Both positive"
        else:
            note = "Tanh negative, Sig positive"
        
        print(f"{x:5.0f} | {tanh_out:6.3f} | {sig_out:7.3f} | {note}")
    
    print("\nKey differences:")
    print("- Tanh is zero-centered, sigmoid is not")
    print("- Tanh range: (-1,1), sigmoid range: (0,1)")
    print("- Tanh has stronger gradients around zero")
    
    # Example 3: Gradient analysis
    print("\n" + "="*50)
    print("Example 3: Gradient Analysis")
    
    test_points = np.array([-3, -1, 0, 1, 3])
    
    print("Input | Output | Gradient | Max Grad | Notes")
    print("-" * 55)
    
    for x in test_points:
        output = tanh(x)
        grad = tanh.backward(np.array([1.0]))[0]
        
        if abs(x) > 2:
            note = "Vanishing gradient"
        elif abs(x) < 0.5:
            note = "Strong gradient"
        else:
            note = "Moderate gradient"
        
        print(f"{x:5.0f} | {output:6.3f} | {grad:8.4f} | {1.0:8.4f} | {note}")
    
    print("Maximum gradient at x=0: 1.0 (vs sigmoid's 0.25)")
    print("Gradient formula: 1 - tanh²(x)")
    
    # Example 4: Vanishing gradient demonstration
    print("\n" + "="*50)
    print("Example 4: Vanishing Gradient Problem")
    
    # Show gradient behavior across range
    x_range = np.array([-5, -3, -1, 0, 1, 3, 5])
    
    print("Demonstrating gradient flow:")
    print("Input | Tanh Out | Tanh Grad | Sigmoid Grad | Better?")
    print("-" * 60)
    
    for x in x_range:
        tanh_out = tanh(x)
        tanh_grad = tanh.backward(np.array([1.0]))[0]
        
        # Calculate sigmoid gradient for comparison
        sig_out = sigmoid(x)
        sig_grad = sigmoid.backward(np.array([1.0]))[0]
        
        better = "Yes" if tanh_grad > sig_grad else "No"
        
        print(f"{x:5.0f} | {tanh_out:8.4f} | {tanh_grad:9.4f} | {sig_grad:12.4f} | {better:>6s}")
    
    print("\nInsights:")
    print("- Tanh has stronger gradients than sigmoid")
    print("- But still vanishes for |x| > 3")
    print("- Better than sigmoid but not good enough for deep networks")
    
    # Example 5: Practical usage
    print("\n" + "="*50)
    print("Example 5: When to Use Tanh")
    
    print("✅ Good for:")
    print("  - Hidden layers in shallow networks")
    print("  - RNN/LSTM gates")
    print("  - When you need zero-centered outputs")
    print("  - Replacing sigmoid for better gradients")
    
    print("\n❌ Avoid for:")
    print("  - Deep neural networks (use ReLU)")
    print("  - When you need only positive outputs")
    print("  - Computational efficiency critical applications")
    
    print("\n🔄 Relationship to sigmoid:")
    print("  tanh(x) = 2*sigmoid(2x) - 1")
    print("  This shows tanh is a shifted and scaled sigmoid")
    
    # Demonstrate the relationship
    x_test = 1.0
    tanh_val = tanh(x_test)
    sig_val = sigmoid(2 * x_test)
    relationship_val = 2 * sig_val - 1
    
    print(f"\nVerification for x={x_test}:")
    print(f"  tanh({x_test}) = {tanh_val:.6f}")
    print(f"  2*sigmoid(2*{x_test}) - 1 = {relationship_val:.6f}")
    print(f"  Match: {np.isclose(tanh_val, relationship_val)}")
    
    print("\n" + "="*50)
    print("Summary: Tanh vs Sigmoid vs ReLU")
    print("Tanh:    Zero-centered, stronger gradients than sigmoid, still vanishes")
    print("Sigmoid: Not zero-centered, weak gradients, good for output layer")
    print("ReLU:    Solves vanishing gradient, computationally efficient, modern choice")
