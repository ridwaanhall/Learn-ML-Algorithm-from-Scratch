# Mathematical Foundations for Machine Learning from Scratch

This guide explains the essential mathematics behind each algorithm in our project. Understanding these foundations will help you truly master machine learning rather than just using it as a black box.

## 📚 Prerequisites

### Essential Math Skills
- **Linear Algebra**: Vectors, matrices, dot products
- **Calculus**: Derivatives, partial derivatives, chain rule
- **Statistics**: Mean, variance, probability distributions
- **Basic Programming**: NumPy operations

Don't worry if you're rusty - we'll review each concept as needed!

## 🎯 Linear Regression Deep Dive

### The Problem
Given input features X and target values y, find the best linear relationship:
```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

In matrix form: `y = Xw + b`

### The Mathematics

#### 1. Forward Pass (Prediction)
```python
# Mathematical formula
y_pred = X @ weights + bias

# What this means:
# For each sample i: y_pred[i] = sum(X[i,j] * weights[j]) + bias
```

#### 2. Loss Function (Mean Squared Error)
```python
# Mathematical formula
loss = (1/n) * sum((y_true - y_pred)²)

# Why squared?
# - Penalizes large errors more
# - Always positive
# - Differentiable everywhere
# - Convex (has global minimum)
```

#### 3. Gradient Calculation
The key insight: **gradients tell us how to change weights to reduce loss**

```python
# Partial derivatives (calculated using chain rule)
∂loss/∂weights = -(2/n) * X.T @ (y_true - y_pred)
∂loss/∂bias = -(2/n) * sum(y_true - y_pred)

# Why negative gradient?
# Gradient points uphill, we want to go downhill
```

#### 4. Parameter Update
```python
# Gradient Descent update rule
weights = weights - learning_rate * gradient_weights
bias = bias - learning_rate * gradient_bias

# Why subtract?
# Move opposite to gradient direction (downhill)
```

### Step-by-Step Derivation

#### Deriving the Gradient for Weights
Starting with loss function:
```
L = (1/n) * Σᵢ(yᵢ - ŷᵢ)²
```

Where ŷᵢ = Σⱼ(xᵢⱼ * wⱼ) + b

Taking partial derivative with respect to wⱼ:
```
∂L/∂wⱼ = (1/n) * Σᵢ 2(yᵢ - ŷᵢ) * (-∂ŷᵢ/∂wⱼ)
        = (1/n) * Σᵢ 2(yᵢ - ŷᵢ) * (-xᵢⱼ)
        = -(2/n) * Σᵢ(yᵢ - ŷᵢ) * xᵢⱼ
```

In matrix form: `∂L/∂w = -(2/n) * X.T @ (y - ŷ)`

## 🧠 Activation Functions Mathematics

### ReLU (Rectified Linear Unit)
```python
# Forward pass
f(x) = max(0, x)

# Derivative
f'(x) = 1 if x > 0 else 0

# Why useful?
# - Simple computation
# - Sparse activation (many zeros)
# - No vanishing gradient for positive inputs
```

### Sigmoid
```python
# Forward pass
f(x) = 1 / (1 + e^(-x))

# Derivative (elegant property!)
f'(x) = f(x) * (1 - f(x))

# Properties:
# - Output range: (0, 1) - good for probabilities
# - Smooth and differentiable
# - Problem: vanishing gradients for |x| >> 0
```

### Tanh
```python
# Forward pass
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

# Derivative
f'(x) = 1 - f(x)²

# Properties:
# - Output range: (-1, 1) - zero-centered
# - Still suffers from vanishing gradients
```

### Softmax (for multi-class classification)
```python
# Forward pass
f(xᵢ) = e^(xᵢ) / Σⱼ e^(xⱼ)

# Properties:
# - Outputs sum to 1 (probability distribution)
# - Differentiable
# - Emphasizes largest input (winner-take-most)
```

## 🎯 Loss Functions Deep Dive

### Mean Squared Error (MSE)
```python
# Formula
MSE = (1/n) * Σᵢ(yᵢ - ŷᵢ)²

# Gradient
∂MSE/∂ŷ = -(2/n) * (y - ŷ)

# When to use:
# - Regression problems
# - When large errors should be penalized heavily
# - When you want smooth gradients
```

### Cross Entropy Loss
```python
# Binary Cross Entropy
BCE = -(1/n) * Σᵢ[yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

# Categorical Cross Entropy
CCE = -(1/n) * Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)

# Why logarithm?
# - Heavily penalizes confident wrong predictions
# - Connects to maximum likelihood estimation
# - Natural for probability outputs
```

### Mean Absolute Error (MAE)
```python
# Formula
MAE = (1/n) * Σᵢ|yᵢ - ŷᵢ|

# Properties:
# - More robust to outliers than MSE
# - Not differentiable at zero (but still usable)
# - All errors weighted equally
```

## ⚡ Optimization Algorithms Mathematics

### Gradient Descent
```python
# Update rule
θₜ₊₁ = θₜ - α * ∇f(θₜ)

# Where:
# θ = parameters (weights, bias)
# α = learning rate
# ∇f = gradient of loss function

# Convergence:
# - Guaranteed for convex functions with appropriate α
# - α too large: oscillation or divergence
# - α too small: slow convergence
```

### Adam (Adaptive Moment Estimation)
```python
# Maintains moving averages of gradients and squared gradients
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t        # momentum
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²       # adaptive learning rate

# Bias correction
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

# Update rule
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)

# Advantages:
# - Adaptive learning rates per parameter
# - Works well with sparse gradients
# - Less sensitive to hyperparameter choices
```

### RMSprop
```python
# Maintains moving average of squared gradients
v_t = γ * v_{t-1} + (1 - γ) * g_t²

# Update rule
θ_t = θ_{t-1} - α * g_t / (√v_t + ε)

# Key insight:
# - Divides learning rate by running average of gradient magnitudes
# - Automatically adapts to gradient scale
```

### Momentum
```python
# Accumulates velocity in consistent gradient directions
v_t = γ * v_{t-1} + α * g_t

# Update rule
θ_t = θ_{t-1} - v_t

# Physics analogy:
# - Ball rolling down hill gains momentum
# - Helps escape local minima
# - Smooths out noisy gradients
```

## 🔍 Key Mathematical Insights

### Why Gradients Work
1. **Local Linear Approximation**: Near any point, a smooth function is approximately linear
2. **Steepest Descent**: Gradient points in direction of steepest increase
3. **Iterative Improvement**: Small steps in right direction eventually reach minimum

### The Chain Rule in Deep Learning
For function composition f(g(x)):
```
df/dx = (df/dg) * (dg/dx)
```

This allows us to compute gradients through multiple layers:
```python
# Forward: x → hidden → output
# Backward: ∂loss/∂output → ∂loss/∂hidden → ∂loss/∂x
```

### Convexity and Global Minima
- **Convex functions**: Have single global minimum (like MSE for linear regression)
- **Non-convex functions**: Have multiple local minima (like neural networks)
- **Implication**: Linear models have guaranteed convergence, neural networks don't

## 📊 Practical Implications

### Learning Rate Selection
- **Too high**: `loss = NaN` or oscillating loss
- **Too low**: Very slow convergence
- **Just right**: Steady decrease in loss
- **Adaptive optimizers**: Automatically adjust learning rates

### Initialization Matters
- **Zero initialization**: Symmetry problem (all neurons learn same thing)
- **Random initialization**: Breaks symmetry
- **Xavier/He initialization**: Maintains gradient scale through layers

### Batch Size Effects
- **Full batch**: Stable gradients, slow updates
- **Mini-batch**: Balance between stability and speed
- **Stochastic (batch=1)**: Noisy but can escape local minima

## 🧮 Working Through Examples

### Example 1: Manual Gradient Calculation
Given: X = [[1, 2], [3, 4]], y = [3, 7], weights = [0.5, 0.5], bias = 0

1. **Forward pass**:
   ```
   y_pred = X @ weights + bias = [1*0.5 + 2*0.5, 3*0.5 + 4*0.5] + 0 = [1.5, 3.5]
   ```

2. **Loss calculation**:
   ```
   loss = 0.5 * ((3-1.5)² + (7-3.5)²) = 0.5 * (2.25 + 12.25) = 7.25
   ```

3. **Gradient calculation**:
   ```
   errors = y - y_pred = [1.5, 3.5]
   grad_weights = -X.T @ errors / n = -[[1,3],[2,4]] @ [1.5,3.5] / 2 = [-6, -9]
   grad_bias = -sum(errors) / n = -5/2 = -2.5
   ```

4. **Parameter update** (lr=0.1):
   ```
   new_weights = [0.5, 0.5] - 0.1 * [-6, -9] = [1.1, 1.4]
   new_bias = 0 - 0.1 * (-2.5) = 0.25
   ```

### Example 2: Understanding Activation Functions
For input x = [-2, -1, 0, 1, 2]:

- **ReLU**: [0, 0, 0, 1, 2] - sparse, preserves positive values
- **Sigmoid**: [0.12, 0.27, 0.5, 0.73, 0.88] - squashes to (0,1)
- **Tanh**: [-0.96, -0.76, 0, 0.76, 0.96] - zero-centered

## 💡 Tips for Mathematical Understanding

1. **Start with simple examples**: Work through calculations by hand
2. **Visualize**: Plot functions, gradients, and loss landscapes
3. **Verify numerically**: Compare analytical gradients with numerical approximations
4. **Build intuition**: Understand what each mathematical operation accomplishes
5. **Connect to code**: See how mathematical formulas become NumPy operations

## 🔗 Mathematical Connections

### Linear Algebra ↔ Machine Learning
- **Matrix multiplication**: Efficient computation of linear transformations
- **Vectorization**: Process multiple samples simultaneously
- **Eigenvalues/eigenvectors**: PCA, understanding optimization landscapes

### Calculus ↔ Optimization
- **Derivatives**: Local rates of change
- **Gradients**: Multi-dimensional derivatives
- **Second derivatives**: Curvature information (used in Newton's method)

### Probability ↔ Loss Functions
- **Maximum likelihood**: Minimizing negative log-likelihood
- **Cross entropy**: Information-theoretic interpretation
- **Bayesian inference**: Principled uncertainty quantification

## 🎯 Next Steps

1. **Practice derivations**: Work through gradient calculations for each algorithm
2. **Implement from scratch**: Start with mathematical formulas, not our code
3. **Verify understanding**: Compare your implementations with ours
4. **Explore extensions**: Add regularization, different optimizers, etc.
5. **Visualize everything**: Plot loss surfaces, decision boundaries, gradients

Remember: Mathematics is the language of machine learning. Understanding it deeply will make you a much more effective practitioner! 🚀
