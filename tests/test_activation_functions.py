"""
Unit tests for activation functions.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.activation_functions.relu import ReLU
from src.activation_functions.sigmoid import Sigmoid
from src.activation_functions.tanh import Tanh
from src.activation_functions.softmax import Softmax

def test_relu():
    """Test ReLU activation function."""
    relu = ReLU()
    x = np.array([-2, -1, 0, 1, 2])
    
    # Test forward pass
    output = relu.forward(x)
    expected = np.array([0, 0, 0, 1, 2])
    assert np.allclose(output, expected), f"Expected {expected}, got {output}"
    
    # Test backward pass
    grad_output = np.ones_like(x)
    grad_input = relu.backward(grad_output)
    expected_grad = np.array([0, 0, 0, 1, 1])
    assert np.allclose(grad_input, expected_grad), "ReLU gradient calculation failed"
    
    print("✓ ReLU tests passed")

def test_sigmoid():
    """Test Sigmoid activation function."""
    sigmoid = Sigmoid()
    x = np.array([-2, -1, 0, 1, 2])
    
    # Test forward pass
    output = sigmoid.forward(x)
    # Sigmoid output should be between 0 and 1
    assert np.all(output >= 0) and np.all(output <= 1), "Sigmoid output should be in [0, 1]"
    
    # Test backward pass
    grad_output = np.ones_like(x)
    grad_input = sigmoid.backward(grad_output)
    assert grad_input.shape == x.shape, "Gradient shape mismatch"
    
    print("✓ Sigmoid tests passed")

def test_tanh():
    """Test Tanh activation function."""
    tanh = Tanh()
    x = np.array([-2, -1, 0, 1, 2])
    
    # Test forward pass
    output = tanh.forward(x)
    # Tanh output should be between -1 and 1
    assert np.all(output >= -1) and np.all(output <= 1), "Tanh output should be in [-1, 1]"
    
    # Test backward pass
    grad_output = np.ones_like(x)
    grad_input = tanh.backward(grad_output)
    assert grad_input.shape == x.shape, "Gradient shape mismatch"
    
    print("✓ Tanh tests passed")

def test_softmax():
    """Test Softmax activation function."""
    softmax = Softmax()
    x = np.array([[1, 2, 3], [4, 5, 6]])
    
    # Test forward pass
    output = softmax.forward(x)
    # Softmax output should sum to 1 for each row
    row_sums = np.sum(output, axis=1)
    assert np.allclose(row_sums, 1), "Softmax rows should sum to 1"
    
    print("✓ Softmax tests passed")

if __name__ == "__main__":
    test_relu()
    test_sigmoid()
    test_tanh()
    test_softmax()
    print("All activation function tests passed!")
