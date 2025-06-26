"""
Loss Functions Module

This module contains implementations of various loss functions used in machine learning.
Each loss function includes:
- Mathematical formula explanation
- Forward pass implementation
- Backward pass (gradient) implementation
- Use case recommendations

Loss functions determine how we measure the difference between predicted and actual values.
"""

from .mse import MSE
from .cross_entropy import CrossEntropy
from .mae import MAE
from .huber import HuberLoss

__all__ = ['MSE', 'CrossEntropy', 'MAE', 'HuberLoss']
