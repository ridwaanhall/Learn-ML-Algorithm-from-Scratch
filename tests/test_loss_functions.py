"""
Unit tests for loss functions.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.loss_functions.mse import MSE
from src.loss_functions.mae import MAE
from src.loss_functions.cross_entropy import CrossEntropy
from src.loss_functions.huber import HuberLoss

def test_mse():
    """Test Mean Squared Error loss function."""
    mse = MSE()
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 2.1, 2.9])
    
    loss = mse.forward(y_true, y_pred)
    expected_loss = np.mean((y_true - y_pred) ** 2)
    
    assert np.allclose(loss, expected_loss), f"Expected {expected_loss}, got {loss}"
    
    # Test gradient
    grad = mse.backward(y_true, y_pred)
    expected_grad = -2 * (y_true - y_pred) / len(y_true)
    
    assert np.allclose(grad, expected_grad), "MSE gradient calculation failed"
    print("✓ MSE tests passed")

def test_mae():
    """Test Mean Absolute Error loss function."""
    mae = MAE()
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 2.1, 2.9])
    
    loss = mae.forward(y_true, y_pred)
    expected_loss = np.mean(np.abs(y_true - y_pred))
    
    assert np.allclose(loss, expected_loss), f"Expected {expected_loss}, got {loss}"
    print("✓ MAE tests passed")

def test_cross_entropy():
    """Test Cross Entropy loss function."""
    ce = CrossEntropy()
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0.1, 0.8, 0.1])
    
    loss = ce.forward(y_true, y_pred)
    # Cross entropy should be positive
    assert loss > 0, "Cross entropy loss should be positive"
    print("✓ Cross Entropy tests passed")

def test_huber():
    """Test Huber loss function."""
    huber = HuberLoss(delta=1.0)
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 2.1, 2.9])
    
    loss = huber.forward(y_true, y_pred)
    assert loss >= 0, "Huber loss should be non-negative"
    print("✓ Huber tests passed")

if __name__ == "__main__":
    test_mse()
    test_mae()
    test_cross_entropy()
    test_huber()
    print("All loss function tests passed!")
