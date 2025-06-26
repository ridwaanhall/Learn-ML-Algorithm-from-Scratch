"""
Momentum optimizer implementation.

This module implements the Momentum optimization algorithm, which helps
accelerate gradients in the relevant direction and dampens oscillations.
It does this by adding a fraction of the update vector of the past time step
to the current update vector.
"""
import numpy as np

class Momentum:
    """
    Momentum optimizer implementation.
    
    The momentum method accumulates an exponentially decaying moving average
    of past gradients and continues to move in their direction.
    
    Update rule:
    v_t = γ * v_{t-1} + η * ∇f(θ)
    θ_t = θ_{t-1} - v_t
    
    where:
    - v_t is the velocity at time t
    - γ is the momentum coefficient (typically 0.9)
    - η is the learning rate
    - ∇f(θ) is the gradient
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Initialize Momentum optimizer.
        
        Args:
            learning_rate (float): Learning rate for parameter updates
            momentum (float): Momentum coefficient (0 ≤ momentum < 1)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.t = 0  # time step
    
    def update(self, parameters, gradients):
        """
        Update parameters using momentum.
        
        Args:
            parameters (np.ndarray): Current parameter values
            gradients (np.ndarray): Gradients with respect to parameters
            
        Returns:
            np.ndarray: Updated parameters
        """
        self.t += 1
        
        # Initialize velocity on first update
        if self.velocity is None:
            self.velocity = np.zeros_like(parameters)
        
        # Update velocity
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradients
        
        # Update parameters
        updated_parameters = parameters - self.velocity
        
        return updated_parameters
    
    def reset(self):
        """Reset optimizer state."""
        self.velocity = None
        self.t = 0

class NesterovMomentum:
    """
    Nesterov Accelerated Gradient (NAG) optimizer implementation.
    
    Nesterov momentum is a variant of momentum that applies the gradient
    at the anticipated future position rather than the current position.
    
    Update rule:
    v_t = γ * v_{t-1} + η * ∇f(θ_{t-1} - γ * v_{t-1})
    θ_t = θ_{t-1} - v_t
    
    This can be reformulated as:
    v_t = γ * v_{t-1} + η * ∇f(θ)
    θ_t = θ_{t-1} - γ * v_t - η * ∇f(θ)
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Initialize Nesterov Momentum optimizer.
        
        Args:
            learning_rate (float): Learning rate for parameter updates
            momentum (float): Momentum coefficient (0 ≤ momentum < 1)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.t = 0  # time step
    
    def update(self, parameters, gradients):
        """
        Update parameters using Nesterov momentum.
        
        Args:
            parameters (np.ndarray): Current parameter values
            gradients (np.ndarray): Gradients with respect to parameters
            
        Returns:
            np.ndarray: Updated parameters
        """
        self.t += 1
        
        # Initialize velocity on first update
        if self.velocity is None:
            self.velocity = np.zeros_like(parameters)
        
        # Update velocity
        velocity_prev = self.velocity.copy()
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradients
        
        # Nesterov update
        updated_parameters = parameters - self.momentum * velocity_prev - self.learning_rate * gradients
        
        return updated_parameters
    
    def reset(self):
        """Reset optimizer state."""
        self.velocity = None
        self.t = 0

class MomentumScheduler:
    """
    Momentum scheduler that adjusts momentum during training.
    """
    
    def __init__(self, optimizer, schedule_type='constant', **kwargs):
        """
        Initialize momentum scheduler.
        
        Args:
            optimizer: Momentum optimizer instance
            schedule_type (str): Type of schedule ('constant', 'linear', 'exponential')
            **kwargs: Additional parameters for the schedule
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.initial_momentum = optimizer.momentum
        self.kwargs = kwargs
    
    def step(self, epoch):
        """
        Update momentum based on the schedule.
        
        Args:
            epoch (int): Current epoch number
        """
        if self.schedule_type == 'constant':
            # No change
            pass
        elif self.schedule_type == 'linear':
            # Linear increase to target momentum
            target_momentum = self.kwargs.get('target_momentum', 0.99)
            total_epochs = self.kwargs.get('total_epochs', 100)
            
            progress = min(epoch / total_epochs, 1.0)
            self.optimizer.momentum = (self.initial_momentum + 
                                     progress * (target_momentum - self.initial_momentum))
        
        elif self.schedule_type == 'exponential':
            # Exponential approach to target momentum
            target_momentum = self.kwargs.get('target_momentum', 0.99)
            rate = self.kwargs.get('rate', 0.1)
            
            self.optimizer.momentum = target_momentum - (target_momentum - self.initial_momentum) * np.exp(-rate * epoch)
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
