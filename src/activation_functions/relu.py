"""
ReLU (Rectified Linear Unit) Activation Function

Mathematical Formula:
ReLU(x) = max(0, x)

Derivative:
ReLU'(x) = 1 if x > 0, else 0

Characteristics:
- Range: [0, +∞)
- Non-saturating for positive inputs
- Sparse activation (many neurons output 0)
- Computationally efficient
- Solves vanishing gradient problem for positive inputs

Use Cases:
- Hidden layers in deep neural networks (most popular choice)
- When you want sparse representations
- When training deep networks (avoids vanishing gradients)
- Default choice for most hidden layers

When NOT to use:
- Output layer for regression (unbounded output)
- When you need negative outputs
- Small networks where dead neurons might be problematic
- When input data is mostly negative

Pros:
- Computationally efficient (simple max operation)
- Reduces vanishing gradient problem
- Promotes sparsity
- Works well in practice

Cons:
- Dead neurons (neurons that always output 0)
- Not zero-centered
- Can suffer from "dying ReLU" problem
"""

import numpy as np


class ReLU:
    """ReLU Activation Function
    
    The most popular activation function for hidden layers in neural networks.
    Simple, effective, and computationally efficient.
    """
    
    def __init__(self):
        """Initialize ReLU activation function"""
        self.name = "ReLU"
        
    def forward(self, x):
        """
        Apply ReLU activation function
        
        Args:
            x (np.ndarray): Input values of any shape
            
        Returns:
            np.ndarray: ReLU activated values, same shape as input
            
        Mathematical Steps:
        1. For each element x_i in input:
           - If x_i > 0: output x_i
           - If x_i ≤ 0: output 0
        2. This is equivalent to: max(0, x_i)
        """
        # Ensure input is numpy array
        x = np.array(x)
        
        # Apply ReLU: max(0, x)
        # This sets all negative values to 0, keeps positive values unchanged
        output = np.maximum(0, x)
        
        # Store input for backward pass (needed for gradient calculation)
        self.last_input = x
        
        return output
    
    def backward(self, grad_output):
        """
        Calculate the gradient of ReLU
        
        Args:
            grad_output (np.ndarray): Gradient from the next layer
            
        Returns:
            np.ndarray: Gradient with respect to input
            
        Mathematical Derivation:
        ReLU(x) = max(0, x)
        
        ∂ReLU/∂x = 1 if x > 0
                 = 0 if x ≤ 0
        
        This means:
        - Gradient flows through unchanged for positive inputs
        - Gradient is blocked (set to 0) for negative inputs
        """
        if not hasattr(self, 'last_input'):
            raise ValueError("Must call forward() before backward()")
        
        # Create gradient mask: 1 where input > 0, 0 elsewhere
        grad_mask = (self.last_input > 0).astype(float)
        
        # Apply mask to gradient
        grad_input = grad_output * grad_mask
        
        return grad_input
    
    def __call__(self, x):
        """Allow the class to be called as a function"""
        return self.forward(x)
    
    def __str__(self):
        return "ReLU: f(x) = max(0, x)"
    
    def __repr__(self):
        return "ReLU()"


class LeakyReLU:
    """Leaky ReLU Activation Function
    
    A variant of ReLU that allows small negative values instead of zero.
    This helps alleviate the "dying ReLU" problem.
    
    Mathematical Formula:
    LeakyReLU(x) = x if x > 0, else α*x
    
    Where α is a small positive constant (typically 0.01)
    """
    
    def __init__(self, alpha=0.01):
        """
        Initialize Leaky ReLU activation function
        
        Args:
            alpha (float): Slope for negative inputs (typically 0.01)
        """
        self.alpha = alpha
        self.name = f"LeakyReLU(α={alpha})"
        
    def forward(self, x):
        """Apply Leaky ReLU activation function"""
        x = np.array(x)
        
        # LeakyReLU: x if x > 0, else α*x
        output = np.where(x > 0, x, self.alpha * x)
        
        # Store input for backward pass
        self.last_input = x
        
        return output
    
    def backward(self, grad_output):
        """Calculate the gradient of Leaky ReLU"""
        if not hasattr(self, 'last_input'):
            raise ValueError("Must call forward() before backward()")
        
        # Gradient: 1 if x > 0, else α
        grad_mask = np.where(self.last_input > 0, 1.0, self.alpha)
        grad_input = grad_output * grad_mask
        
        return grad_input
    
    def __call__(self, x):
        return self.forward(x)
    
    def __str__(self):
        return f"LeakyReLU: f(x) = x if x > 0, else {self.alpha}*x"


# Example usage and educational demonstration
if __name__ == "__main__":
    # Create ReLU activation function
    relu = ReLU()
    leaky_relu = LeakyReLU(alpha=0.01)
    
    # Test with various inputs
    test_inputs = [
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
        np.array([[-1, 2], [3, -4]]),  # 2D array
        np.array([0.5, -0.5, 10, -10])
    ]
    
    print("=== ReLU Activation Function Examples ===\n")
    
    for i, x in enumerate(test_inputs, 1):
        print(f"Example {i}:")
        print(f"Input: {x}")
        
        # Forward pass
        relu_output = relu(x)
        leaky_output = leaky_relu(x)
        
        print(f"ReLU output: {relu_output}")
        print(f"Leaky ReLU output: {leaky_output}")
        
        # Backward pass (using dummy gradient)
        dummy_grad = np.ones_like(x)
        relu_grad = relu.backward(dummy_grad)
        leaky_grad = leaky_relu.backward(dummy_grad)
        
        print(f"ReLU gradient: {relu_grad}")
        print(f"Leaky ReLU gradient: {leaky_grad}")
        print(f"Note: Gradient is 1 for positive inputs, 0 (or α) for negative\n")
    
    # Demonstrate the "dead neuron" problem
    print("=== Demonstrating Dead Neuron Problem ===")
    # Simulate a neuron that receives large negative inputs
    large_negative_input = np.array([-100, -50, -10])
    
    relu_output = relu(large_negative_input)
    dummy_grad = np.array([1.0, 1.0, 1.0])
    relu_grad = relu.backward(dummy_grad)
    
    print(f"Large negative inputs: {large_negative_input}")
    print(f"ReLU output: {relu_output}")
    print(f"ReLU gradient: {relu_grad}")
    print("Note: All outputs are 0, and gradients are 0 - this neuron is 'dead'")
    print("It cannot recover because no gradient flows through it.")
    
    # Show how Leaky ReLU helps
    leaky_output = leaky_relu(large_negative_input)
    leaky_grad = leaky_relu.backward(dummy_grad)
    
    print(f"\nLeaky ReLU output: {leaky_output}")
    print(f"Leaky ReLU gradient: {leaky_grad}")
    print("Note: Small negative outputs and gradients allow the neuron to potentially recover")
