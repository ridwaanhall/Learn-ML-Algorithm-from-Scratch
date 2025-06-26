"""
RMSprop (Root Mean Square propagation) Optimizer

Mathematical Formula:
v_t = β * v_{t-1} + (1 - β) * g_t²
θ_{t+1} = θ_t - α * g_t / (√v_t + ε)

Where:
- θ = parameters
- g_t = gradient at time t
- v_t = moving average of squared gradients
- α = learning rate
- β = decay rate (typically 0.9)
- ε = small constant for numerical stability

Use Cases:
- Non-stationary objectives (changing loss landscape)
- RNN training (where gradients can vary widely)
- When you want adaptive learning rates per parameter
- Alternative to Adam when momentum is not needed

When NOT to use:
- Simple convex problems (SGD might be sufficient)
- When you need momentum (use Adam instead)
- Very sparse gradients (Adam handles better)

Advantages:
- Adaptive learning rates per parameter
- Good for non-stationary objectives
- Less memory than Adam (no momentum term)
- Often converges faster than basic SGD

Disadvantages:
- Can still get stuck in local minima
- Learning rate can become too small over time
- No momentum to help escape local minima
"""

import numpy as np


class RMSprop:
    """RMSprop Optimizer
    
    Adapts the learning rate for each parameter based on the moving average
    of squared gradients. Good for non-stationary objectives.
    """
    
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        """
        Initialize RMSprop optimizer
        
        Args:
            learning_rate (float): Step size for parameter updates
            rho (float): Decay rate for moving average of squared gradients
            epsilon (float): Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        
        # Internal state
        self.v = {}  # Moving average of squared gradients
        
        self.name = "RMSprop"
        
    def update(self, params, gradients):
        """
        Update parameters using RMSprop algorithm
        
        Args:
            params (dict): Dictionary of parameter names to values
            gradients (dict): Dictionary of parameter names to gradients
            
        Returns:
            dict: Updated parameters
        """
        updated_params = {}
        
        for param_name, param_value in params.items():
            if param_name not in gradients:
                # If no gradient for this parameter, don't update it
                updated_params[param_name] = param_value
                continue
                
            grad = gradients[param_name]
            
            # Initialize moving average if first time
            if param_name not in self.v:
                self.v[param_name] = np.zeros_like(param_value)
            
            # Update moving average of squared gradients
            # v_t = ρ * v_{t-1} + (1 - ρ) * g_t²
            self.v[param_name] = self.rho * self.v[param_name] + (1 - self.rho) * (grad ** 2)
            
            # Update parameters
            # θ_{t+1} = θ_t - α * g_t / (√v_t + ε)
            param_update = self.learning_rate * grad / (np.sqrt(self.v[param_name]) + self.epsilon)
            updated_params[param_name] = param_value - param_update
            
        return updated_params
    
    def reset(self):
        """Reset optimizer state (useful for new training runs)"""
        self.v = {}
    
    def get_state(self):
        """Get current optimizer state"""
        return {
            'learning_rate': self.learning_rate,
            'rho': self.rho,
            'epsilon': self.epsilon,
            'v': self.v.copy()
        }
    
    def __str__(self):
        return f"RMSprop(lr={self.learning_rate}, ρ={self.rho}, ε={self.epsilon})"
    
    def __repr__(self):
        return (f"RMSprop(learning_rate={self.learning_rate}, rho={self.rho}, "
                f"epsilon={self.epsilon})")


# Educational demonstration and examples
if __name__ == "__main__":
    # Create RMSprop optimizer
    rmsprop = RMSprop(learning_rate=0.01, rho=0.9)
    
    print("=== RMSprop Optimizer Educational Examples ===\n")
    
    # Example 1: Basic parameter update
    print("Example 1: Basic Parameter Update")
    
    # Initial parameters
    params = {
        'weights': np.array([1.0, 2.0]),
        'bias': np.array([0.5])
    }
    
    # Simulated gradients
    gradients = {
        'weights': np.array([0.1, -0.2]),
        'bias': np.array([0.05])
    }
    
    print("Initial parameters:")
    for name, value in params.items():
        print(f"  {name}: {value}")
    
    print("\nGradients:")
    for name, value in gradients.items():
        print(f"  {name}: {value}")
    
    # Update parameters
    updated_params = rmsprop.update(params, gradients)
    
    print("\nUpdated parameters:")
    for name, value in updated_params.items():
        print(f"  {name}: {value}")
    
    print("\nMoving averages of squared gradients:")
    for name, value in rmsprop.v.items():
        print(f"  v[{name}]: {value}")
    
    # Example 2: Multiple updates showing adaptation
    print("\n" + "="*50)
    print("Example 2: Adaptive Learning Rate Demonstration")
    
    # Reset optimizer
    rmsprop.reset()
    
    # Simulate parameter with changing gradient magnitudes
    current_param = np.array([1.0])
    gradient_sequence = [
        np.array([0.1]),   # Small gradient
        np.array([0.1]),   # Small gradient
        np.array([1.0]),   # Large gradient (sudden change)
        np.array([1.0]),   # Large gradient
        np.array([0.1]),   # Back to small gradient
    ]
    
    print("Step | Gradient | v (sq grad avg) | Effective LR | New Param")
    print("-" * 65)
    
    for step, grad in enumerate(gradient_sequence, 1):
        params = {'param': current_param.copy()}
        grads = {'param': grad}
        
        # Update
        updated = rmsprop.update(params, grads)
        current_param = updated['param']
        
        # Calculate effective learning rate
        v_val = rmsprop.v['param'][0]
        effective_lr = rmsprop.learning_rate / (np.sqrt(v_val) + rmsprop.epsilon)
        
        print(f"{step:4d} | {grad[0]:8.3f} | {v_val:13.6f} | {effective_lr[0]:11.6f} | {current_param[0]:9.6f}")
    
    print("\nObservations:")
    print("- Large gradients → larger v → smaller effective learning rate")
    print("- Small gradients → smaller v → larger effective learning rate")
    print("- This prevents overshooting with large gradients")
    
    # Example 3: Comparison with SGD
    print("\n" + "="*50)
    print("Example 3: RMSprop vs SGD on Oscillating Gradients")
    
    from .gradient_descent import GradientDescent
    
    # Create optimizers
    rmsprop_comp = RMSprop(learning_rate=0.01)
    sgd_comp = GradientDescent(learning_rate=0.01)
    
    # Simulate oscillating gradients (common problem)
    oscillating_grads = [
        np.array([1.0]),    # Large positive
        np.array([-0.8]),   # Large negative
        np.array([0.6]),    # Medium positive
        np.array([-0.4]),   # Medium negative
        np.array([0.2]),    # Small positive
    ]
    
    # Track both optimizers
    param_rmsprop = np.array([1.0])
    param_sgd = np.array([1.0])
    
    print("Step | Gradient | RMSprop Param | SGD Param | RMSprop Change | SGD Change")
    print("-" * 80)
    
    for step, grad in enumerate(oscillating_grads, 1):
        # RMSprop update
        params_rms = {'param': param_rmsprop.copy()}
        grads = {'param': grad}
        updated_rms = rmsprop_comp.update(params_rms, grads)
        new_param_rms = updated_rms['param']
        change_rms = new_param_rms - param_rmsprop
        
        # SGD update
        params_sgd = {'param': param_sgd.copy()}
        updated_sgd = sgd_comp.update(params_sgd, grads)
        new_param_sgd = updated_sgd['param']
        change_sgd = new_param_sgd - param_sgd
        
        print(f"{step:4d} | {grad[0]:8.3f} | {new_param_rms[0]:13.6f} | {new_param_sgd[0]:9.6f} | "
              f"{change_rms[0]:14.6f} | {change_sgd[0]:10.6f}")
        
        param_rmsprop = new_param_rms
        param_sgd = new_param_sgd
    
    print("\nObservation: RMSprop adapts step size based on gradient history")
    
    # Example 4: Different decay rates
    print("\n" + "="*50)
    print("Example 4: Effect of Decay Rate (rho)")
    
    decay_rates = [0.5, 0.9, 0.99]
    test_gradients = [np.array([1.0]), np.array([0.1]), np.array([1.0])]
    
    print("Testing different rho values with gradient sequence: [1.0, 0.1, 1.0]")
    print()
    
    for rho in decay_rates:
        print(f"Rho = {rho}:")
        optimizer = RMSprop(learning_rate=0.01, rho=rho)
        param = np.array([1.0])
        
        for step, grad in enumerate(test_gradients, 1):
            params = {'param': param.copy()}
            grads = {'param': grad}
            updated = optimizer.update(params, grads)
            param = updated['param']
            
            v_val = optimizer.v['param'][0]
            print(f"  Step {step}: grad={grad[0]:.1f}, v={v_val:.6f}, param={param[0]:.6f}")
        print()
    
    print("Insights:")
    print("- Lower rho: Faster adaptation to recent gradients")
    print("- Higher rho: More stable, considers longer history")
    print("- rho=0.9 is typically a good default")
    
    print("\n" + "="*50)
    print("Summary: When to Use RMSprop")
    print("✅ Non-stationary objectives")
    print("✅ RNN training")
    print("✅ When gradients vary widely in magnitude")
    print("✅ Alternative to Adam without momentum")
    print("❌ Simple convex problems")
    print("❌ When you need momentum effects")
    print("⚙️  Key hyperparameter: rho (decay rate, default 0.9)")
