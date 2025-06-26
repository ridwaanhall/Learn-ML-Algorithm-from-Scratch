"""
Adam (Adaptive Moment Estimation) Optimizer

Mathematical Formula:
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²

m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

θ_{t+1} = θ_t - α * m̂_t / (√v̂_t + ε)

Where:
- θ = parameters
- g_t = gradient at time t
- m_t = 1st moment estimate (momentum)
- v_t = 2nd moment estimate (squared gradients)
- α = learning rate
- β₁, β₂ = decay rates for moments
- ε = small constant for numerical stability

Use Cases:
- Most deep learning tasks (default choice)
- When you have noisy gradients
- When you want adaptive learning rates per parameter
- Large datasets with mini-batch training
- When you don't want to tune learning rate extensively

When NOT to use:
- Simple linear models (SGD might be sufficient)
- When you have very limited data
- When you need guaranteed convergence (use SGD with proper schedule)
- When memory is extremely limited

Advantages:
- Adaptive learning rates for each parameter
- Works well with default hyperparameters
- Handles sparse gradients well
- Relatively robust to hyperparameter choices

Disadvantages:
- More memory usage (stores momentum and velocity)
- Can sometimes fail to converge to optimal solution
- More complex than basic SGD
"""

import numpy as np


class Adam:
    """Adam Optimizer
    
    Combines the advantages of AdaGrad and RMSprop with momentum.
    Maintains moving averages of both gradients and squared gradients.
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer
        
        Args:
            learning_rate (float): Step size for parameter updates (α)
            beta1 (float): Decay rate for 1st moment estimate (momentum)
            beta2 (float): Decay rate for 2nd moment estimate (squared gradients)
            epsilon (float): Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Internal state
        self.m = {}  # 1st moment estimates
        self.v = {}  # 2nd moment estimates
        self.t = 0   # Time step
        
        self.name = "Adam"
        
    def update(self, params, gradients):
        """
        Update parameters using Adam algorithm
        
        Args:
            params (dict): Dictionary of parameter names to values
            gradients (dict): Dictionary of parameter names to gradients
            
        Returns:
            dict: Updated parameters
        """
        # Increment time step
        self.t += 1
        
        updated_params = {}
        
        for param_name, param_value in params.items():
            if param_name not in gradients:
                # If no gradient for this parameter, don't update it
                updated_params[param_name] = param_value
                continue
                
            grad = gradients[param_name]
            
            # Initialize moment estimates if first time
            if param_name not in self.m:
                self.m[param_name] = np.zeros_like(param_value)
                self.v[param_name] = np.zeros_like(param_value)
            
            # Update biased first moment estimate
            # m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate  
            # v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            # m̂_t = m_t / (1 - β₁^t)
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            # v̂_t = v_t / (1 - β₂^t)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            # θ_{t+1} = θ_t - α * m̂_t / (√v̂_t + ε)
            param_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_params[param_name] = param_value - param_update
            
        return updated_params
    
    def reset(self):
        """Reset optimizer state (useful for new training runs)"""
        self.m = {}
        self.v = {}
        self.t = 0
    
    def get_state(self):
        """Get current optimizer state"""
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            't': self.t,
            'm': self.m.copy(),
            'v': self.v.copy()
        }
    
    def __str__(self):
        return f"Adam(lr={self.learning_rate}, β₁={self.beta1}, β₂={self.beta2}, ε={self.epsilon})"
    
    def __repr__(self):
        return (f"Adam(learning_rate={self.learning_rate}, beta1={self.beta1}, "
                f"beta2={self.beta2}, epsilon={self.epsilon})")


# Educational demonstration and examples
if __name__ == "__main__":
    # Create Adam optimizer
    adam = Adam(learning_rate=0.01, beta1=0.9, beta2=0.999)
    
    print("=== Adam Optimizer Educational Examples ===\n")
    
    # Example 1: Simple parameter update
    print("Example 1: Basic Parameter Update")
    
    # Initial parameters
    params = {
        'weights': np.array([1.0, 2.0, 3.0]),
        'bias': np.array([0.5])
    }
    
    # Simulated gradients
    gradients = {
        'weights': np.array([0.1, -0.2, 0.3]),
        'bias': np.array([0.05])
    }
    
    print("Initial parameters:")
    for name, value in params.items():
        print(f"  {name}: {value}")
    
    print("\nGradients:")
    for name, value in gradients.items():
        print(f"  {name}: {value}")
    
    # Update parameters
    updated_params = adam.update(params, gradients)
    
    print("\nUpdated parameters:")
    for name, value in updated_params.items():
        print(f"  {name}: {value}")
    
    print(f"\nOptimizer state after update: t={adam.t}")
    print("First moment estimates (m):")
    for name, value in adam.m.items():
        print(f"  m[{name}]: {value}")
    print("Second moment estimates (v):")
    for name, value in adam.v.items():
        print(f"  v[{name}]: {value}")
    
    # Example 2: Multiple updates to show adaptation
    print("\n" + "="*50)
    print("Example 2: Multiple Updates Showing Adaptation")
    
    # Reset optimizer
    adam.reset()
    
    # Simulate training with changing gradients
    training_data = [
        {'weights': np.array([0.1, -0.2, 0.3]), 'bias': np.array([0.05])},  # Step 1
        {'weights': np.array([0.2, -0.1, 0.2]), 'bias': np.array([0.03])},  # Step 2
        {'weights': np.array([0.05, -0.3, 0.4]), 'bias': np.array([0.07])}, # Step 3
        {'weights': np.array([0.15, -0.15, 0.1]), 'bias': np.array([0.02])}, # Step 4
    ]
    
    current_params = {
        'weights': np.array([1.0, 2.0, 3.0]),
        'bias': np.array([0.5])
    }
    
    print("Training progression:")
    print(f"Initial: weights={current_params['weights']}, bias={current_params['bias']}")
    
    for step, grads in enumerate(training_data, 1):
        current_params = adam.update(current_params, grads)
        print(f"Step {step}: weights={current_params['weights']:.4f}, bias={current_params['bias']:.4f}")
    
    # Example 3: Comparing with different hyperparameters
    print("\n" + "="*50)
    print("Example 3: Hyperparameter Comparison")
    
    # Different Adam configurations
    adam_configs = [
        Adam(learning_rate=0.001, beta1=0.9, beta2=0.999),  # Default
        Adam(learning_rate=0.01, beta1=0.9, beta2=0.999),   # Higher LR
        Adam(learning_rate=0.001, beta1=0.5, beta2=0.999),  # Lower momentum
        Adam(learning_rate=0.001, beta1=0.9, beta2=0.9),    # Lower RMSprop factor
    ]
    
    config_names = ["Default", "High LR", "Low Momentum", "Low RMSprop"]
    
    # Single gradient step for comparison
    test_params = {'weight': np.array([1.0])}
    test_grad = {'weight': np.array([0.1])}
    
    print("Single update with gradient [0.1] on parameter [1.0]:")
    
    for config, name in zip(adam_configs, config_names):
        updated = config.update(test_params.copy(), test_grad)
        change = updated['weight'][0] - test_params['weight'][0]
        print(f"{name:15s}: {updated['weight'][0]:.6f} (change: {change:+.6f})")
    
    print("\nKey Insights:")
    print("- Higher learning rate → larger parameter changes")
    print("- Lower β₁ (momentum) → less smoothing of gradients")
    print("- Lower β₂ (RMSprop) → less adaptation based on gradient magnitude")
    print("- Adam automatically adapts the effective learning rate per parameter")
