"""
Unit tests for optimization algorithms.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.optimization.gradient_descent import GradientDescent
from src.optimization.adam import Adam
from src.optimization.rmsprop import RMSprop

def test_gradient_descent():
    """Test Gradient Descent optimizer."""
    optimizer = GradientDescent(learning_rate=0.01)
    
    # Test parameter initialization
    params = {'weights': np.array([1.0, 2.0, 3.0]), 'bias': np.array([0.5])}
    gradients = {'weights': np.array([0.1, 0.2, 0.3]), 'bias': np.array([0.1])}
    
    # Test update
    updated_params = optimizer.update(params, gradients)
    expected_weights = params['weights'] - 0.01 * gradients['weights']
    expected_bias = params['bias'] - 0.01 * gradients['bias']
    
    assert np.allclose(updated_params['weights'], expected_weights), "Gradient descent weights update failed"
    assert np.allclose(updated_params['bias'], expected_bias), "Gradient descent bias update failed"
    print("✓ Gradient Descent tests passed")

def test_adam():
    """Test Adam optimizer."""
    optimizer = Adam(learning_rate=0.001)
    
    # Test parameter initialization
    params = {'weights': np.array([1.0, 2.0, 3.0]), 'bias': np.array([0.5])}
    gradients = {'weights': np.array([0.1, 0.2, 0.3]), 'bias': np.array([0.1])}
    
    # Test multiple updates
    original_weights = params['weights'].copy()
    for _ in range(3):
        updated_params = optimizer.update(params, gradients)
        params = updated_params
    
    # Adam should update parameters
    assert not np.allclose(params['weights'], original_weights), "Adam should update parameters"
    print("✓ Adam tests passed")

def test_rmsprop():
    """Test RMSprop optimizer."""
    optimizer = RMSprop(learning_rate=0.001)
    
    # Test parameter initialization
    params = {'weights': np.array([1.0, 2.0, 3.0]), 'bias': np.array([0.5])}
    gradients = {'weights': np.array([0.1, 0.2, 0.3]), 'bias': np.array([0.1])}
    
    # Test multiple updates
    original_weights = params['weights'].copy()
    for _ in range(3):
        updated_params = optimizer.update(params, gradients)
        params = updated_params
    
    # RMSprop should update parameters
    assert not np.allclose(params['weights'], original_weights), "RMSprop should update parameters"
    print("✓ RMSprop tests passed")

if __name__ == "__main__":
    test_gradient_descent()
    test_adam()
    test_rmsprop()
    print("All optimization tests passed!")
