"""
Stochastic Gradient Descent (SGD) optimizer
"""
import numpy as np
from typing import Dict, Any, Optional


class SGDOptimizer:
    """
    Stochastic Gradient Descent optimizer
    
    Updates parameters using the gradient of the loss function:
    θ = θ - learning_rate * gradient
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize SGD optimizer
        
        Args:
            learning_rate: Learning rate for parameter updates
            momentum: Momentum factor (0 means no momentum)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.initialized = False
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Update parameters using SGD with optional momentum
        
        Args:
            params: Current parameters
            gradients: Gradients with respect to parameters
            
        Returns:
            Updated parameters
        """
        if not self.initialized:
            self.velocity = np.zeros_like(params)
            self.initialized = True
        
        # Update velocity with momentum
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradients
        
        # Update parameters
        updated_params = params - self.velocity
        
        return updated_params
    
    def reset(self):
        """Reset optimizer state"""
        self.velocity = None
        self.initialized = False
    
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration"""
        return {
            'learning_rate': self.learning_rate,
            'momentum': self.momentum
        }


class SGDScheduler:
    """
    Learning rate scheduler for SGD
    """
    
    def __init__(self, initial_lr: float = 0.01, schedule_type: str = 'constant', **kwargs):
        """
        Initialize learning rate scheduler
        
        Args:
            initial_lr: Initial learning rate
            schedule_type: Type of schedule ('constant', 'step', 'exponential')
            **kwargs: Additional parameters for specific schedules
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.schedule_type = schedule_type
        self.kwargs = kwargs
        self.step_count = 0
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.current_lr
    
    def step(self) -> None:
        """Update learning rate based on schedule"""
        self.step_count += 1
        
        if self.schedule_type == 'constant':
            pass  # No change
        elif self.schedule_type == 'step':
            step_size = self.kwargs.get('step_size', 100)
            gamma = self.kwargs.get('gamma', 0.1)
            if self.step_count % step_size == 0:
                self.current_lr *= gamma
        elif self.schedule_type == 'exponential':
            gamma = self.kwargs.get('gamma', 0.99)
            self.current_lr = self.initial_lr * (gamma ** self.step_count)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def reset(self):
        """Reset scheduler state"""
        self.current_lr = self.initial_lr
        self.step_count = 0
