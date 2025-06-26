# Troubleshooting Guide

This comprehensive guide helps you diagnose and fix common issues when implementing machine learning algorithms from scratch.

## 🚨 Common Training Problems

### Problem: Model Not Learning (Loss Not Decreasing)

**Symptoms:**
- Loss stays constant or decreases very slowly
- Training accuracy remains at random levels
- Model predictions don't improve

**Diagnosis Steps:**

1. **Check Learning Rate**
   ```python
   # Try reducing learning rate by 10x
   optimizer = GradientDescent(learning_rate=0.001)  # instead of 0.01
   ```

2. **Verify Data Preprocessing**
   ```python
   # Check for proper scaling
   print(f"Feature means: {np.mean(X, axis=0)}")
   print(f"Feature stds: {np.std(X, axis=0)}")
   
   # Features should have similar scales
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

3. **Examine Gradient Flow**
   ```python
   # Check if gradients are too small
   loss_fn = MSE()
   predictions = model.predict(X)
   gradients = loss_fn.backward(y, predictions)
   print(f"Gradient magnitude: {np.linalg.norm(gradients)}")
   
   # Should be > 1e-6 typically
   ```

**Solutions:**
- ✅ **Reduce learning rate**: Try 0.001 instead of 0.01
- ✅ **Scale your features**: Use StandardScaler or MinMaxScaler
- ✅ **Increase epochs**: More iterations for convergence
- ✅ **Check data quality**: Remove NaN values, outliers
- ✅ **Try different optimizer**: Adam instead of SGD

### Problem: Loss Becomes NaN or Infinite

**Symptoms:**
- Loss suddenly jumps to `nan` or `inf`
- Model parameters become extremely large
- Predictions are all zeros or very large numbers

**Diagnosis:**
```python
# Check for exploding gradients
if np.isnan(loss) or np.isinf(loss):
    print("Numerical instability detected!")
    print(f"Max parameter value: {np.max(np.abs(model.weights))}")
    print(f"Max gradient value: {np.max(np.abs(gradients))}")
```

**Solutions:**
- ✅ **Reduce learning rate dramatically**: Try 0.0001
- ✅ **Gradient clipping**: Limit gradient magnitude
  ```python
  # Clip gradients to prevent explosion
  max_grad_norm = 1.0
  grad_norm = np.linalg.norm(gradients)
  if grad_norm > max_grad_norm:
      gradients = gradients * (max_grad_norm / grad_norm)
  ```
- ✅ **Check input data**: Look for extreme values or NaN
- ✅ **Use more stable optimizer**: Adam handles this better than SGD

### Problem: Model Overfitting

**Symptoms:**
- Training loss decreases but validation loss increases
- Perfect training accuracy but poor test accuracy
- Model memorizes training data

**Diagnosis:**
```python
# Monitor train vs validation performance
train_loss = loss_fn.forward(y_train, model.predict(X_train))
val_loss = loss_fn.forward(y_val, model.predict(X_val))
print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

# Large gap indicates overfitting
```

**Solutions:**
- ✅ **Reduce model complexity**: Fewer features or simpler model
- ✅ **Get more training data**: If possible
- ✅ **Early stopping**: Stop when validation loss starts increasing
- ✅ **Add regularization**: L1 or L2 penalties (advanced topic)

### Problem: Model Underfitting

**Symptoms:**
- Both training and validation loss are high
- Model performs poorly on all datasets
- Loss plateaus quickly at high value

**Solutions:**
- ✅ **Increase model complexity**: Add polynomial features
- ✅ **Feature engineering**: Create more informative features
- ✅ **Check for sufficient data**: Need enough samples for learning
- ✅ **Verify model choice**: Linear model might not fit non-linear data

## 🔧 Data-Related Issues

### Problem: Poor Data Quality

**Symptoms:**
- Inconsistent training behavior
- Model performs worse than expected
- Unexpected predictions

**Diagnostic Code:**
```python
# Check for missing values
print(f"NaN count: {np.isnan(X).sum()}")
print(f"Infinite values: {np.isinf(X).sum()}")

# Check data distribution
print(f"Feature ranges: {np.ptp(X, axis=0)}")  # Peak-to-peak
print(f"Target distribution: {np.histogram(y, bins=10)}")

# Look for outliers
Q1 = np.percentile(X, 25, axis=0)
Q3 = np.percentile(X, 75, axis=0)
IQR = Q3 - Q1
outliers = np.any((X < Q1 - 1.5*IQR) | (X > Q3 + 1.5*IQR), axis=1)
print(f"Outlier percentage: {outliers.mean()*100:.1f}%")
```

**Solutions:**
- ✅ **Handle missing values**: Remove or impute
- ✅ **Remove/treat outliers**: Use robust loss functions like Huber
- ✅ **Scale features**: Standardize or normalize
- ✅ **Feature selection**: Remove irrelevant features

### Problem: Insufficient Training Data

**Symptoms:**
- High variance in results
- Performance depends heavily on train/test split
- Model doesn't generalize well

**Solutions:**
- ✅ **Data augmentation**: Create synthetic samples (carefully)
- ✅ **Simpler model**: Reduce complexity to match data amount
- ✅ **Cross-validation**: Get better performance estimates
- ✅ **Transfer learning**: Use pre-trained features (advanced)

## ⚙️ Implementation Bugs

### Problem: Incorrect Gradient Calculation

**Symptoms:**
- Model learns very slowly or not at all
- Loss behaves unexpectedly
- Debugging shows gradients are wrong

**Verification:**
```python
# Numerical gradient check
def numerical_gradient(f, x, h=1e-5):
    """Calculate numerical gradient for verification."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Compare analytical vs numerical gradients
analytical_grad = your_gradient_function(...)
numerical_grad = numerical_gradient(your_loss_function, parameters)
difference = np.abs(analytical_grad - numerical_grad)
print(f"Max gradient difference: {np.max(difference)}")
# Should be < 1e-5 for correct implementation
```

**Solutions:**
- ✅ **Double-check math**: Verify derivative calculations
- ✅ **Check matrix dimensions**: Ensure proper broadcasting
- ✅ **Test with simple examples**: Use known analytical solutions

### Problem: Shape Mismatches

**Symptoms:**
- `ValueError` about incompatible shapes
- Broadcasting errors
- Unexpected output dimensions

**Debugging:**
```python
# Add shape checking throughout your code
print(f"X shape: {X.shape}")
print(f"weights shape: {weights.shape}")
print(f"predictions shape: {predictions.shape}")
print(f"y shape: {y.shape}")

# Verify matrix multiplication compatibility
assert X.shape[1] == weights.shape[0], f"Incompatible: {X.shape} @ {weights.shape}"
```

**Solutions:**
- ✅ **Add assertions**: Check shapes at key points
- ✅ **Use explicit reshaping**: `y.reshape(-1, 1)` if needed
- ✅ **Understand broadcasting**: Learn NumPy broadcasting rules

## 📊 Performance Issues

### Problem: Slow Convergence

**Symptoms:**
- Model takes many epochs to converge
- Loss decreases very slowly
- Training time is excessive

**Solutions:**
- ✅ **Use adaptive optimizers**: Adam, RMSprop instead of SGD
- ✅ **Feature scaling**: Standardize all features
- ✅ **Better initialization**: Xavier or He initialization
- ✅ **Learning rate scheduling**: Start high, reduce over time
- ✅ **Batch processing**: Mini-batch instead of full batch

### Problem: Memory Issues

**Symptoms:**
- `MemoryError` during training
- System becomes unresponsive
- Out of memory errors

**Solutions:**
- ✅ **Mini-batch training**: Process smaller chunks of data
- ✅ **Data streaming**: Load data in batches from disk
- ✅ **Reduce precision**: Use float32 instead of float64
- ✅ **Memory profiling**: Identify memory leaks

## 🎯 Hyperparameter Issues

### Problem: How to Choose Learning Rate

**Method 1: Learning Rate Range Test**
```python
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
losses = []

for lr in learning_rates:
    model = LinearRegression()
    optimizer = GradientDescent(learning_rate=lr)
    history = model.fit(X, y, loss_fn, optimizer, epochs=100, verbose=False)
    losses.append(history['losses'][-1])
    
# Plot losses vs learning rates
plt.semilogx(learning_rates, losses)
plt.xlabel('Learning Rate')
plt.ylabel('Final Loss')
plt.title('Learning Rate Range Test')
```

**Method 2: Monitor Training Curves**
```python
# Good learning rate: Smooth, steady decrease
# Too high: Oscillating or increasing loss
# Too low: Very slow decrease
```

### Problem: Choosing the Right Optimizer

**Guidelines:**
- **Start with Adam**: Good default choice
- **Use SGD for simple problems**: Linear regression with clean data
- **Try RMSprop for RNNs**: Better for sequential data
- **Momentum for smooth landscapes**: When you have consistent gradients

**Comparison Code:**
```python
optimizers = {
    'SGD': GradientDescent(learning_rate=0.01),
    'Adam': Adam(learning_rate=0.001),
    'RMSprop': RMSprop(learning_rate=0.001)
}

results = {}
for name, opt in optimizers.items():
    model = LinearRegression()
    history = model.fit(X, y, MSE(), opt, epochs=1000, verbose=False)
    results[name] = history['losses'][-1]
    
print("Final losses:", results)
```

## 🔍 Debugging Strategies

### Systematic Debugging Process

1. **Start Simple**
   ```python
   # Test with tiny dataset first
   X_tiny = X[:10]
   y_tiny = y[:10]
   
   # Should overfit perfectly with enough epochs
   model.fit(X_tiny, y_tiny, loss_fn, optimizer, epochs=1000)
   assert loss < 1e-6, "Model should overfit small dataset"
   ```

2. **Check Individual Components**
   ```python
   # Test loss function
   loss_fn = MSE()
   test_loss = loss_fn.forward(np.array([1, 2]), np.array([1.1, 1.9]))
   expected = 0.01  # (0.1² + 0.1²) / 2
   assert np.isclose(test_loss, expected), "Loss calculation incorrect"
   
   # Test optimizer
   params = {'w': np.array([1.0])}
   grads = {'w': np.array([0.1])}
   new_params = optimizer.update(params, grads)
   # Verify update is correct
   ```

3. **Visualize Everything**
   ```python
   # Plot loss curves
   plt.plot(history['losses'])
   plt.yscale('log')
   plt.title('Training Loss')
   
   # Plot predictions vs actual
   plt.scatter(y_test, predictions)
   plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
   plt.xlabel('Actual')
   plt.ylabel('Predicted')
   
   # Plot residuals
   residuals = y_test - predictions
   plt.scatter(predictions, residuals)
   plt.axhline(y=0, color='r', linestyle='--')
   ```

### Advanced Debugging Techniques

**Gradient Checking:**
```python
def gradient_check(model, X, y, loss_fn, epsilon=1e-7):
    """Verify gradients are correct using finite differences."""
    
    # Get analytical gradients
    predictions = model.predict(X)
    loss = loss_fn.forward(y, predictions)
    analytical_grads = loss_fn.backward(y, predictions)
    
    # Compute numerical gradients
    numerical_grads = np.zeros_like(model.weights)
    for i in range(len(model.weights)):
        model.weights[i] += epsilon
        loss_plus = loss_fn.forward(y, model.predict(X))
        
        model.weights[i] -= 2 * epsilon
        loss_minus = loss_fn.forward(y, model.predict(X))
        
        numerical_grads[i] = (loss_plus - loss_minus) / (2 * epsilon)
        model.weights[i] += epsilon  # restore
    
    # Compare
    relative_error = np.abs(analytical_grads - numerical_grads) / (np.abs(analytical_grads) + np.abs(numerical_grads) + 1e-8)
    print(f"Max relative error: {np.max(relative_error)}")
    return relative_error < 1e-5
```

## 📋 Quick Reference Checklist

### Before Training
- [ ] Data is clean (no NaN, extreme outliers handled)
- [ ] Features are scaled appropriately
- [ ] Train/validation split is representative
- [ ] Model architecture makes sense for the problem

### During Training
- [ ] Loss is decreasing (not oscillating or constant)
- [ ] Gradients are reasonable magnitude (not too small/large)
- [ ] No NaN or infinite values in loss/parameters
- [ ] Validation performance tracks training performance

### After Training
- [ ] Model generalizes to test data
- [ ] Predictions make intuitive sense
- [ ] Performance metrics are reasonable
- [ ] Residuals don't show obvious patterns

### Red Flags 🚩
- Loss suddenly jumps to NaN
- Gradients are all zeros
- Training accuracy is 100% but test accuracy is random
- Loss oscillates wildly
- Model predicts same value for all inputs

## 💡 Pro Tips

1. **Always start with a simple baseline**: Random predictions, simple linear model
2. **Test on synthetic data first**: Generate data where you know the true answer
3. **Use version control**: Git helps track what changes broke things
4. **Print intermediate values**: Don't be afraid to add debug prints
5. **Compare with known implementations**: Verify against scikit-learn when possible
6. **Read error messages carefully**: Python error messages often point to the exact issue

Remember: Debugging is a skill that improves with practice. Most issues are caused by simple mistakes in data preprocessing, learning rates, or mathematical errors! 🚀
