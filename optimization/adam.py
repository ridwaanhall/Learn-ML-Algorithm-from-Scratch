"""
Adam optimizer implementation
"""
import numpy as np
from typing import Dict, Any


class AdamOptimizer:
    """
    Adam (Adaptive Moment Estimation) optimizer
    
    Combines the advantages of AdaGrad and RMSProp:
    - Maintains moving averages of gradient (momentum)
    - Maintains moving averages of squared gradient (adaptive learning rate)
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize Adam optimizer
        
        Args:
            learning_rate: Learning rate (step size)
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates  
            epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # State variables
        self.m = None  # First moment (momentum)
        self.v = None  # Second moment (squared gradients)
        self.t = 0     # Time step
        self.initialized = False
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Update parameters using Adam optimization
        
        Args:
            params: Current parameters
            gradients: Gradients with respect to parameters
            
        Returns:
            Updated parameters
        """
        if not self.initialized:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            self.initialized = True
        
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        updated_params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params
    
    def reset(self):
        """Reset optimizer state"""
        self.m = None
        self.v = None
        self.t = 0
        self.initialized = False
    
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration"""
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon
        }


class AdamWOptimizer(AdamOptimizer):
    """
    AdamW optimizer - Adam with decoupled weight decay
    
    AdamW modifies the weight decay in Adam to be decoupled from the gradient.
    This often leads to better generalization.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.01):
        """
        Initialize AdamW optimizer
        
        Args:
            learning_rate: Learning rate (step size)
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay coefficient
        """
        super().__init__(learning_rate, beta1, beta2, epsilon)
        self.weight_decay = weight_decay
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Update parameters using AdamW optimization
        
        Args:
            params: Current parameters
            gradients: Gradients with respect to parameters
            
        Returns:
            Updated parameters
        """
        if not self.initialized:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            self.initialized = True
        
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters with decoupled weight decay
        updated_params = params - self.learning_rate * (
            m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * params
        )
        
        return updated_params
    
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration"""
        config = super().get_config()
        config['weight_decay'] = self.weight_decay
        return config
