"""
Gradient Descent Optimizer

Mathematical Formula:
θ_{t+1} = θ_t - α * ∇L(θ_t)

Where:
- θ = parameters (weights, bias)
- α = learning rate
- ∇L = gradient of loss function
- t = time step

Variants:
1. Batch Gradient Descent: Uses entire dataset
2. Stochastic Gradient Descent (SGD): Uses single sample
3. Mini-batch Gradient Descent: Uses subset of data

Use Cases:
- Simple optimization problems
- When you want to understand optimization fundamentals
- Convex optimization problems
- When memory is limited
- As baseline comparison for other optimizers

When NOT to use:
- Non-convex problems with many local minima
- When gradients are very noisy
- When you need fast convergence
- Very large datasets (unless using mini-batches)

Advantages:
- Simple to understand and implement
- Guaranteed convergence for convex functions with proper learning rate
- Low memory usage
- Good for educational purposes

Disadvantages:
- Can get stuck in local minima
- Sensitive to learning rate choice
- Slow convergence compared to adaptive methods
- Struggles with noisy gradients
"""

import numpy as np


class GradientDescent:
    """Gradient Descent Optimizer
    
    The most basic optimization algorithm. Updates parameters by moving
    in the direction opposite to the gradient (steepest descent).
    """
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize Gradient Descent optimizer
        
        Args:
            learning_rate (float): Step size for parameter updates
                - Too large: May overshoot minimum, cause divergence
                - Too small: Slow convergence
                - Typical range: 0.001 to 0.1
        """
        self.learning_rate = learning_rate
        self.name = "Gradient Descent"
        
        # Track optimization history
        self.history = []
        
    def update(self, params, gradients):
        """
        Update parameters using gradient descent
        
        Args:
            params (dict): Dictionary of parameter names to values
            gradients (dict): Dictionary of parameter names to gradients
            
        Returns:
            dict: Updated parameters
            
        Mathematical Steps:
        1. For each parameter θ: θ_new = θ_old - α * ∇θ
        2. Where α is learning rate and ∇θ is gradient
        """
        updated_params = {}
        update_info = {}
        
        for param_name, param_value in params.items():
            if param_name not in gradients:
                # If no gradient for this parameter, don't update it
                updated_params[param_name] = param_value
                continue
            
            grad = gradients[param_name]
            
            # Gradient descent update: θ = θ - α * ∇θ
            param_update = self.learning_rate * grad
            updated_params[param_name] = param_value - param_update
            
            # Store update information for analysis
            update_info[param_name] = {
                'gradient': grad,
                'update': param_update,
                'old_value': param_value,
                'new_value': updated_params[param_name]
            }
        
        # Store history
        self.history.append(update_info)
        
        return updated_params
    
    def reset(self):
        """Reset optimizer state"""
        self.history = []
    
    def get_learning_rate(self):
        """Get current learning rate"""
        return self.learning_rate
    
    def set_learning_rate(self, new_lr):
        """Set new learning rate"""
        self.learning_rate = new_lr
    
    def get_history(self):
        """Get optimization history"""
        return self.history
    
    def __str__(self):
        return f"GradientDescent(learning_rate={self.learning_rate})"
    
    def __repr__(self):
        return f"GradientDescent(learning_rate={self.learning_rate})"


class SGD(GradientDescent):
    """Stochastic Gradient Descent
    
    A variant of gradient descent that uses random samples or mini-batches
    instead of the full dataset. More commonly used in practice.
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.0):
        """
        Initialize SGD optimizer
        
        Args:
            learning_rate (float): Step size for parameter updates
            momentum (float): Momentum factor (0 to 1)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}
        self.name = f"SGD(momentum={momentum})"
    
    def update(self, params, gradients):
        """
        Update parameters using SGD with optional momentum
        
        Mathematical Formula with Momentum:
        v_t = μ * v_{t-1} + α * ∇θ_t
        θ_{t+1} = θ_t - v_t
        
        Where μ is momentum coefficient
        """
        updated_params = {}
        
        for param_name, param_value in params.items():
            if param_name not in gradients:
                updated_params[param_name] = param_value
                continue
            
            grad = gradients[param_name]
            
            # Initialize velocity if first time
            if param_name not in self.velocity:
                self.velocity[param_name] = np.zeros_like(param_value)
            
            if self.momentum > 0:
                # Update velocity with momentum
                self.velocity[param_name] = (self.momentum * self.velocity[param_name] + 
                                           self.learning_rate * grad)
                # Update parameters using velocity
                updated_params[param_name] = param_value - self.velocity[param_name]
            else:
                # Standard gradient descent
                updated_params[param_name] = param_value - self.learning_rate * grad
        
        return updated_params
    
    def reset(self):
        """Reset optimizer state"""
        super().reset()
        self.velocity = {}


# Educational demonstration and examples
if __name__ == "__main__":
    print("=== Gradient Descent Optimizer Educational Examples ===\n")
    
    # Example 1: Basic parameter update
    print("Example 1: Basic Parameter Update")
    
    # Create optimizer
    gd = GradientDescent(learning_rate=0.1)
    
    # Simulate parameters and gradients
    params = {
        'weight': np.array([1.0]),
        'bias': np.array([0.5])
    }
    
    gradients = {
        'weight': np.array([0.2]),  # Positive gradient means decrease weight
        'bias': np.array([-0.1])    # Negative gradient means increase bias
    }
    
    print("Initial parameters:")
    for name, value in params.items():
        print(f"  {name}: {value}")
    
    print("Gradients:")
    for name, value in gradients.items():
        print(f"  {name}: {value}")
    
    # Update parameters
    updated_params = gd.update(params, gradients)
    
    print("Updated parameters:")
    for name, value in updated_params.items():
        change = value - params[name]
        print(f"  {name}: {value} (change: {change:+.3f})")
    
    print(f"Learning rate: {gd.learning_rate}")
    print("Note: Parameters move opposite to gradient direction")
    
    # Example 2: Learning rate effects
    print("\n" + "="*50)
    print("Example 2: Effect of Different Learning Rates")
    
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    gradient = np.array([1.0])  # Fixed gradient
    initial_param = np.array([2.0])
    
    print(f"Initial parameter: {initial_param}")
    print(f"Gradient: {gradient}")
    print("\nLearning rate effects:")
    
    for lr in learning_rates:
        optimizer = GradientDescent(learning_rate=lr)
        params = {'param': initial_param.copy()}
        grads = {'param': gradient}
        
        updated = optimizer.update(params, grads)
        change = updated['param'] - initial_param
        
        print(f"LR = {lr:4.2f}: new_value = {updated['param'][0]:6.3f}, "
              f"change = {change[0]:+6.3f}")
    
    print("Observation: Larger learning rates cause bigger parameter changes")
    
    # Example 3: Multi-step optimization
    print("\n" + "="*50)
    print("Example 3: Multi-step Optimization Journey")
    
    # Simulate optimizing f(x) = x², minimum at x = 0
    # Gradient: f'(x) = 2x
    
    optimizer = GradientDescent(learning_rate=0.1)
    current_x = 3.0  # Start at x = 3
    
    print("Optimizing f(x) = x² starting from x = 3.0")
    print("True minimum at x = 0")
    print("\nOptimization steps:")
    print("Step | x value | Gradient | Loss f(x)")
    print("-" * 40)
    
    for step in range(10):
        # Calculate function value and gradient
        loss = current_x ** 2
        gradient = 2 * current_x
        
        print(f"{step:4d} | {current_x:7.4f} | {gradient:8.4f} | {loss:8.4f}")
        
        # Update parameter
        params = {'x': np.array([current_x])}
        grads = {'x': np.array([gradient])}
        updated = optimizer.update(params, grads)
        current_x = updated['x'][0]
    
    print(f"\nFinal x: {current_x:.6f}")
    print("Note: Parameter gradually moves toward the minimum (x = 0)")
    
    # Example 4: SGD with momentum comparison
    print("\n" + "="*50)
    print("Example 4: SGD with Momentum vs Regular GD")
    
    # Compare regular GD with SGD+momentum
    gd_regular = GradientDescent(learning_rate=0.1)
    sgd_momentum = SGD(learning_rate=0.1, momentum=0.9)
    
    # Simulate oscillating gradients (common in practice)
    oscillating_gradients = [1.0, -0.8, 0.6, -0.4, 0.2, -0.1]
    
    print("Simulating optimization with oscillating gradients:")
    print("Gradients:", oscillating_gradients)
    
    # Track both optimizers
    x_gd = 1.0
    x_sgd = 1.0
    
    print("\nStep | Regular GD | SGD+Momentum")
    print("-" * 35)
    print(f"   0 | {x_gd:10.4f} | {x_sgd:11.4f}")
    
    for step, grad in enumerate(oscillating_gradients, 1):
        # Update with regular GD
        params_gd = {'x': np.array([x_gd])}
        grads = {'x': np.array([grad])}
        updated_gd = gd_regular.update(params_gd, grads)
        x_gd = updated_gd['x'][0]
        
        # Update with SGD+momentum
        params_sgd = {'x': np.array([x_sgd])}
        updated_sgd = sgd_momentum.update(params_sgd, grads)
        x_sgd = updated_sgd['x'][0]
        
        print(f"{step:4d} | {x_gd:10.4f} | {x_sgd:11.4f}")
    
    print("\nObservation: Momentum helps smooth out oscillations")
    print("SGD with momentum often converges faster in practice")
