"""
Sigmoid Activation Function

Mathematical Formula:
σ(x) = 1 / (1 + e^(-x))

Derivative:
σ'(x) = σ(x) * (1 - σ(x))

Characteristics:
- Range: (0, 1)
- S-shaped curve
- Smooth and differentiable
- Output can be interpreted as probability
- Saturates at both ends

Use Cases:
- Binary classification output layer
- Gate mechanisms in neural networks
- When you need outputs in (0,1) range
- Probability estimation

When NOT to use:
- Hidden layers in deep networks (vanishing gradient problem)
- When you need outputs that can be negative
- Multi-class classification output (use softmax)

Pros:
- Smooth gradient
- Output range (0,1) useful for probabilities
- Well-understood and stable

Cons:
- Vanishing gradient problem in deep networks
- Not zero-centered (affects convergence)
- Computationally more expensive than ReLU
"""

import numpy as np


class Sigmoid:
    """Sigmoid Activation Function
    
    Classic activation function that maps any real number to (0,1).
    Perfect for binary classification and probability outputs.
    """
    
    def __init__(self):
        """Initialize Sigmoid activation function"""
        self.name = "Sigmoid"
        
    def forward(self, x):
        """
        Apply sigmoid activation function
        
        Args:
            x (np.ndarray): Input values of any shape
            
        Returns:
            np.ndarray: Sigmoid activated values, same shape as input
            
        Mathematical Steps:
        1. For each element x_i: σ(x_i) = 1 / (1 + e^(-x_i))
        2. Uses stable computation to avoid overflow
        """
        # Ensure input is numpy array
        x = np.array(x)
        
        # Stable sigmoid computation to avoid overflow
        # For large positive x: use σ(x) = 1 / (1 + e^(-x))
        # For large negative x: use σ(x) = e^x / (1 + e^x)
        output = np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),           # Stable for x >= 0
            np.exp(x) / (1 + np.exp(x))     # Stable for x < 0
        )
        
        # Store output for backward pass
        self.last_output = output
        
        return output
    
    def backward(self, grad_output):
        """
        Calculate the gradient of sigmoid
        
        Args:
            grad_output (np.ndarray): Gradient from the next layer
            
        Returns:
            np.ndarray: Gradient with respect to input
            
        Mathematical Derivation:
        σ(x) = 1 / (1 + e^(-x))
        
        Let u = 1 + e^(-x), then σ(x) = 1/u
        
        ∂σ/∂x = ∂(1/u)/∂x = -1/u² * ∂u/∂x
              = -1/u² * (-e^(-x))
              = e^(-x) / u²
              = e^(-x) / (1 + e^(-x))²
              = (1/(1 + e^(-x))) * (e^(-x)/(1 + e^(-x)))
              = σ(x) * (1 - σ(x))
        
        This elegant form uses the output of the forward pass!
        """
        if not hasattr(self, 'last_output'):
            raise ValueError("Must call forward() before backward()")
        
        # Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
        sigmoid_derivative = self.last_output * (1 - self.last_output)
        
        # Apply chain rule
        grad_input = grad_output * sigmoid_derivative
        
        return grad_input
    
    def __call__(self, x):
        """Allow the class to be called as a function"""
        return self.forward(x)
    
    def __str__(self):
        return "Sigmoid: σ(x) = 1 / (1 + e^(-x))"
    
    def __repr__(self):
        return "Sigmoid()"


# Example usage and educational demonstration
if __name__ == "__main__":
    # Create sigmoid activation function
    sigmoid = Sigmoid()
    
    print("=== Sigmoid Activation Function Educational Examples ===\n")
    
    # Example 1: Basic behavior
    print("Example 1: Basic Sigmoid Behavior")
    
    # Test with various inputs
    test_inputs = np.array([-10, -2, -1, 0, 1, 2, 10])
    
    print("Input  | Output | Interpretation")
    print("-" * 40)
    
    for x in test_inputs:
        output = sigmoid(x)
        
        if output < 0.1:
            interp = "Very unlikely (close to 0)"
        elif output < 0.4:
            interp = "Unlikely"
        elif output < 0.6:
            interp = "Uncertain (around 0.5)"
        elif output < 0.9:
            interp = "Likely"
        else:
            interp = "Very likely (close to 1)"
        
        print(f"{x:6.0f} | {output:.4f} | {interp}")
    
    print("\nKey observations:")
    print("- Large negative inputs → close to 0")
    print("- Zero input → exactly 0.5")
    print("- Large positive inputs → close to 1")
    print("- S-shaped curve, smooth transition")
    
    # Example 2: Gradient behavior
    print("\n" + "="*50)
    print("Example 2: Sigmoid Gradient Analysis")
    
    # Test gradient at different points
    test_points = np.array([-5, -2, -1, 0, 1, 2, 5])
    
    print("Input | Output | Gradient | Notes")
    print("-" * 50)
    
    for x in test_points:
        output = sigmoid(x)
        
        # Calculate gradient using dummy grad_output = 1
        dummy_grad = np.array([1.0])
        grad = sigmoid.backward(dummy_grad)[0]
        
        if grad < 0.1:
            note = "Vanishing gradient!"
        elif grad > 0.2:
            note = "Strong gradient"
        else:
            note = "Moderate gradient"
        
        print(f"{x:5.0f} | {output:.4f} | {grad:8.4f} | {note}")
    
    print("\nGradient insights:")
    print("- Maximum gradient at x=0 (σ=0.5): 0.25")
    print("- Gradient approaches 0 for |x| > 3")
    print("- This causes vanishing gradient problem in deep networks")
    
    # Example 3: Comparison with other functions
    print("\n" + "="*50)
    print("Example 3: Sigmoid vs Other Activations")
    
    # Compare with tanh and a simple threshold
    x_range = np.array([-3, -1, 0, 1, 3])
    
    print("Input | Sigmoid | Tanh   | Step   | ReLU")
    print("-" * 45)
    
    for x in x_range:
        sig_out = sigmoid(x)
        tanh_out = np.tanh(x)
        step_out = 1.0 if x > 0 else 0.0
        relu_out = max(0, x)
        
        print(f"{x:5.0f} | {sig_out:7.4f} | {tanh_out:6.3f} | {step_out:6.1f} | {relu_out:4.1f}")
    
    # Example 4: Binary classification demonstration
    print("\n" + "="*50)
    print("Example 4: Binary Classification Usage")
    
    # Simulate logits from a binary classifier
    logits = np.array([-2.5, -1.0, -0.1, 0.0, 0.1, 1.0, 2.5])
    probabilities = sigmoid(logits)
    
    print("Logit  | Probability | Prediction | Confidence")
    print("-" * 50)
    
    for logit, prob in zip(logits, probabilities):
        prediction = "Positive" if prob > 0.5 else "Negative"
        confidence = max(prob, 1-prob)
        
        print(f"{logit:6.1f} | {prob:11.4f} | {prediction:10s} | {confidence:.4f}")
    
    print("\nBinary classification insights:")
    print("- Logit 0 → probability 0.5 (uncertain)")
    print("- Positive logits → probability > 0.5")
    print("- Negative logits → probability < 0.5")
    print("- Larger |logit| → more confident prediction")
    
    # Example 5: Vanishing gradient demonstration
    print("\n" + "="*50)
    print("Example 5: Vanishing Gradient Problem")
    
    # Show how gradients diminish with large inputs
    extreme_inputs = np.array([-10, -5, -2, 0, 2, 5, 10])
    
    print("Demonstrating gradient flow through sigmoid:")
    print("Input | Output | Local Grad | If grad_out=1")
    print("-" * 50)
    
    for x in extreme_inputs:
        output = sigmoid(np.array([x]))
        grad = sigmoid.backward(np.array([1.0]))[0]
        
        print(f"{x:5.0f} | {output[0]:6.4f} | {grad:10.6f} | Flows: {grad:.6f}")
    
    print("\nVanishing gradient problem:")
    print("- For |x| > 5: gradient < 0.007")
    print("- In deep networks, gradients multiply through layers")
    print("- Many small gradients → gradient vanishes")
    print("- This is why ReLU is preferred for hidden layers")
    
    print("\n" + "="*50)
    print("Summary: When to use Sigmoid")
    print("✅ Binary classification output layer")
    print("✅ When you need probabilities (0,1)")
    print("✅ Gate mechanisms (LSTM, etc.)")
    print("❌ Hidden layers in deep networks")
    print("❌ When you need negative outputs")
    print("❌ Multi-class problems (use softmax)")
