# Quick Start Guide

Get up and running with machine learning from scratch in 5 minutes!

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Learn-ML-Algorithm-from-Scratch
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import numpy; print('Ready to learn ML!')"
   ```

## 📖 Your First ML Model (2 minutes)

### Step 1: Create your first linear regression

```python
# Import the components
from src.models.linear_regression import LinearRegression
from src.loss_functions.mse import MSE
from src.optimization.gradient_descent import GradientDescent
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])  # Features
y = np.array([2, 4, 6, 8, 10])           # Targets (y = 2*x)

# Create the model components
model = LinearRegression()
loss_fn = MSE()
optimizer = GradientDescent(learning_rate=0.01)

# Train the model
history = model.fit(X, y, loss_fn, optimizer, epochs=100)

# Make predictions
predictions = model.predict([[6], [7]])
print(f"Predictions for x=6,7: {predictions}")
```

### Step 2: Run the complete example

```bash
python examples/complete_linear_regression.py
```

This will show you a full end-to-end example with detailed explanations!

## 🎯 Choose Your Learning Path

### 🌱 **Beginner Path** (Start here!)
1. **Linear Regression** - Learn the basics
   ```bash
   python examples/complete_linear_regression.py
   ```

2. **Understanding Loss Functions**
   ```python
   from src.loss_functions.mse import MSE
   # Run the built-in examples
   python src/loss_functions/mse.py
   ```

3. **Understanding Optimizers**
   ```python
   from src.optimization.gradient_descent import GradientDescent
   # Run the built-in examples  
   python src/optimization/gradient_descent.py
   ```

### 🌿 **Intermediate Path**
1. **Try different optimizers:**
   ```python
   from src.optimization.adam import Adam
   optimizer = Adam(learning_rate=0.001)
   # Compare with gradient descent
   ```

2. **Experiment with activation functions:**
   ```python
   from src.activation_functions.relu import ReLU
   from src.activation_functions.sigmoid import Sigmoid
   # See how they behave differently
   ```

3. **Binary Classification** (Coming soon)

### 🌳 **Advanced Path**
1. **Neural Networks** (Coming soon)
2. **Custom loss functions**
3. **Advanced optimizers**

## 📚 Core Concepts (5-minute read)

### What is Machine Learning?
- **Goal**: Learn patterns from data to make predictions
- **Process**: Adjust model parameters to minimize prediction errors
- **Key**: Find the best balance between fitting data and generalizing to new examples

### The Three Essential Components

1. **Model** (`src/models/`)
   - **What**: Mathematical function that makes predictions
   - **Example**: Linear Regression → `y = weights * x + bias`
   - **Purpose**: Transform input features into predictions

2. **Loss Function** (`src/loss_functions/`)
   - **What**: Measures how wrong your predictions are
   - **Example**: MSE → `loss = (true_value - prediction)²`
   - **Purpose**: Tells the optimizer what to minimize

3. **Optimizer** (`src/optimization/`)
   - **What**: Algorithm that adjusts model parameters
   - **Example**: Gradient Descent → `new_weight = old_weight - learning_rate * gradient`
   - **Purpose**: Iteratively improves the model

### The Training Loop
```python
for epoch in range(num_epochs):
    # 1. Make predictions
    predictions = model.predict(X)
    
    # 2. Calculate loss
    loss = loss_function(y_true, predictions)
    
    # 3. Calculate gradients
    gradients = loss_function.backward(y_true, predictions)
    
    # 4. Update parameters
    model.parameters = optimizer.update(model.parameters, gradients)
```

## 🛠️ Common Recipes

### Recipe 1: Simple Regression
```python
model = LinearRegression()
loss = MSE()
optimizer = GradientDescent(learning_rate=0.01)
# Good for: House prices, temperature prediction, etc.
```

### Recipe 2: Binary Classification
```python
model = LogisticRegression()  # Coming soon
loss = CrossEntropy(binary=True)
optimizer = Adam(learning_rate=0.001)
# Good for: Spam detection, medical diagnosis, etc.
```

### Recipe 3: Multi-class Classification  
```python
model = NeuralNetwork()  # Coming soon
loss = CrossEntropy(binary=False)
optimizer = Adam(learning_rate=0.001)
# Good for: Image classification, text categorization, etc.
```

## 🔧 Quick Troubleshooting

### Problem: Model not learning (loss not decreasing)
**Solutions:**
- ✅ Reduce learning rate by 10x
- ✅ Check if data is properly scaled
- ✅ Increase number of epochs

### Problem: Loss is NaN or infinite
**Solutions:**
- ✅ Reduce learning rate (probably too high)
- ✅ Check for invalid data (NaN, inf)
- ✅ Use gradient clipping

### Problem: Poor predictions
**Solutions:**
- ✅ Try different model (maybe need non-linear)
- ✅ Check if you have enough data
- ✅ Feature engineering
- ✅ Different loss function

## 📖 Next Steps

1. **Read the documentation** → `docs/best_practices.md`
2. **Understand the math** → `docs/math_foundations.md`
3. **Try more examples** → `examples/` directory
4. **Experiment** → Modify the code and see what happens!

## 💡 Learning Tips

1. **Start Simple** - Begin with linear regression before neural networks
2. **Understand Each Component** - Run the individual module examples
3. **Experiment** - Change hyperparameters and observe effects
4. **Read the Code** - Every function has detailed comments
5. **Ask Questions** - Use the code comments to understand the math

## 🎓 What You'll Learn

By the end of this project, you'll understand:

- ✅ How gradient descent actually works
- ✅ Why certain loss functions work for specific problems  
- ✅ When to use different optimizers
- ✅ The mathematical foundations of ML
- ✅ How to implement algorithms from scratch
- ✅ How to debug training problems

## 🤝 Getting Help

- **Read the code comments** - They explain everything!
- **Run the examples** - Each module has educational demonstrations
- **Check the docs** - Comprehensive guides in `docs/`
- **Experiment** - The best way to learn is by doing

## 🎉 You're Ready!

Start with the complete example and work your way through the components. Remember: the goal is understanding, not just getting good results!

```bash
python examples/complete_linear_regression.py
```

Happy learning! 🚀
