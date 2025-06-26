"""
Optimization Algorithms Module

This module contains implementations of various optimization algorithms used to train machine learning models.
Each optimizer includes:
- Mathematical formula explanation
- Step-by-step implementation
- Hyperparameter explanations
- Use case recommendations

Optimizers determine how model parameters are updated based on gradients to minimize the loss function.
"""

from .gradient_descent import GradientDescent
from .adam import Adam
from .rmsprop import RMSprop
from .momentum import Momentum, NesterovMomentum

__all__ = ['GradientDescent', 'Adam', 'RMSprop', 'Momentum', 'NesterovMomentum']
