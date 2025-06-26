"""
Activation Functions Module

This module contains implementations of various activation functions used in neural networks.
Each activation function includes:
- Mathematical formula explanation
- Forward pass implementation
- Derivative implementation for backpropagation
- Use case recommendations and characteristics

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.
"""

from .relu import ReLU
from .sigmoid import Sigmoid
from .tanh import Tanh
from .softmax import Softmax

__all__ = ['ReLU', 'Sigmoid', 'Tanh', 'Softmax']
