"""
Complete Example: Linear Regression from Scratch

This example demonstrates how to use all components together to build and train
a linear regression model from scratch. Perfect for beginners to understand
how different parts of machine learning work together.

Author: Ridwan Hall (ridwaanhall.com)
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt

# Import our custom implementations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.linear_regression import LinearRegression
from src.loss_functions.mse import MSE
from src.optimization.gradient_descent import GradientDescent
from src.optimization.adam import Adam


def generate_sample_data(n_samples=100, noise_level=0.1, random_seed=42):
    """
    Generate synthetic linear data for demonstration
    
    Creates data following: y = 3*x1 + 2*x2 - 1*x3 + 5 + noise
    """
    np.random.seed(random_seed)
    
    # Generate random features
    X = np.random.uniform(-2, 2, (n_samples, 3))
    
    # True parameters we want the model to learn
    true_weights = np.array([3.0, 2.0, -1.0])
    true_bias = 5.0
    
    # Generate target values with noise
    y = np.dot(X, true_weights) + true_bias + np.random.normal(0, noise_level, n_samples)
    
    return X, y, true_weights, true_bias

def scale_features(X):
    """
    Simple feature scaling (standardization)
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    return (X - mean) / std, mean, std

def apply_scaling(X, mean, std):
    """Apply scaling with given mean and std"""
    std = np.where(std == 0, 1, std)
    return (X - mean) / std


def main():
    print("="*60)
    print("COMPLETE LINEAR REGRESSION EXAMPLE")
    print("="*60)
    print()
    
    # Step 1: Generate sample data
    print("Step 1: Generating Sample Data")
    print("-" * 30)
    
    X, y, true_weights, true_bias = generate_sample_data(n_samples=200, noise_level=0.2)
    
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target equation: y = {true_weights[0]}*x1 + {true_weights[1]}*x2 + {true_weights[2]}*x3 + {true_bias}")
    print(f"Feature ranges: X1=[{X[:,0].min():.2f}, {X[:,0].max():.2f}], "
          f"X2=[{X[:,1].min():.2f}, {X[:,1].max():.2f}], "
          f"X3=[{X[:,2].min():.2f}, {X[:,2].max():.2f}]")
    print(f"Target range: y=[{y.min():.2f}, {y.max():.2f}]")
    print()
    
    # Step 2: Split data into train/test
    print("Step 2: Splitting and Scaling Data")
    print("-" * 30)
    
    # Simple train-test split (80-20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features for better convergence
    X_train_scaled, X_mean, X_std = scale_features(X_train)
    X_test_scaled = apply_scaling(X_test, X_mean, X_std)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print("Features scaled for better convergence")
    print()
    
    # Step 3: Create model components
    print("Step 3: Creating Model Components")
    print("-" * 30)
    
    # Create model
    model = LinearRegression()
    print(f"Model: {model}")
    
    # Create loss function
    loss_function = MSE()
    print(f"Loss Function: {loss_function}")
    
    # Create optimizer
    optimizer = GradientDescent(learning_rate=0.1)
    print(f"Optimizer: {optimizer}")
    print()
    
    # Step 4: Train the model
    print("Step 4: Training the Model")
    print("-" * 30)
    
    print("Training in progress...")
    training_history = model.fit(
        X_train_scaled, y_train,
        loss_function=loss_function,
        optimizer=optimizer,
        epochs=2000,
        verbose=False  # Set to True to see training progress
    )
    
    print("Training completed!")
    print(f"Final training loss: {training_history['final_loss']:.6f}")
    print()
    
    # Step 5: Analyze results
    print("Step 5: Analyzing Results")
    print("-" * 30)
    
    print("Learned Parameters:")
    print(f"  Weights: {model.weights}")
    print(f"  Bias: {model.bias:.4f}")
    
    print("True Parameters:")
    print(f"  Weights: {true_weights}")
    print(f"  Bias: {true_bias:.4f}")
    
    print("Parameter Errors:")
    weight_errors = np.abs(model.weights - true_weights)
    bias_error = abs(model.bias - true_bias)
    print(f"  Weight errors: {weight_errors}")
    print(f"  Bias error: {bias_error:.4f}")
    print()
    
    # Step 6: Evaluate on test set
    print("Step 6: Evaluating on Test Set")
    print("-" * 30)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = model.score(X_train_scaled, y_train, metric='r2')
    test_r2 = model.score(X_test_scaled, y_test, metric='r2')
    train_mse = model.score(X_train_scaled, y_train, metric='mse')
    test_mse = model.score(X_test_scaled, y_test, metric='mse')
    
    print("Performance Metrics:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Training MSE: {train_mse:.4f}")
    print(f"  Test MSE: {test_mse:.4f}")
    
    # Check for overfitting
    if train_r2 - test_r2 > 0.1:
        print("  ⚠️  Possible overfitting detected!")
    else:
        print("  ✅ Good generalization!")
    print()
    
    # Step 7: Compare with different optimizer
    print("Step 7: Comparing with Adam Optimizer")
    print("-" * 30)
    
    # Create new model with Adam optimizer
    model_adam = LinearRegression()
    adam_optimizer = Adam(learning_rate=0.01)
    
    print("Training with Adam optimizer...")
    adam_history = model_adam.fit(
        X_train_scaled, y_train,
        loss_function=loss_function,
        optimizer=adam_optimizer,
        epochs=1000,
        verbose=False
    )
    
    adam_test_r2 = model_adam.score(X_test_scaled, y_test, metric='r2')
    
    print("Optimizer Comparison:")
    print(f"  Gradient Descent - Final Loss: {training_history['final_loss']:.6f}, Test R²: {test_r2:.4f}")
    print(f"  Adam             - Final Loss: {adam_history['final_loss']:.6f}, Test R²: {adam_test_r2:.4f}")
    print()
    
    # Step 8: Demonstrate predictions
    print("Step 8: Making New Predictions")
    print("-" * 30)
    
    # Create some example inputs for prediction
    example_inputs = np.array([
        [1.0, 0.5, -0.5],   # Example 1
        [-1.0, 1.5, 0.0],   # Example 2
        [0.0, 0.0, 1.0]     # Example 3
    ])
    
    # Calculate true values
    true_outputs = np.dot(example_inputs, true_weights) + true_bias
    
    # Scale the example inputs using the same scaling as training data
    example_inputs_scaled = apply_scaling(example_inputs, X_mean, X_std)
    
    # Make predictions with our model (using scaled inputs)
    predicted_outputs = model.predict(example_inputs_scaled)
    
    print("Example Predictions:")
    print("Input Features     | True Output | Predicted | Error")
    print("-" * 55)
    for i, (inp, true_out, pred_out) in enumerate(zip(example_inputs, true_outputs, predicted_outputs)):
        error = abs(true_out - pred_out)
        print(f"[{inp[0]:5.1f}, {inp[1]:5.1f}, {inp[2]:5.1f}] | {true_out:9.3f} | {pred_out:9.3f} | {error:.3f}")
    print()
    
    # Step 9: Learning insights
    print("Step 9: Key Learning Insights")
    print("-" * 30)
    
    print("What we learned:")
    print("1. Linear regression finds the best linear relationship between features and target")
    print("2. MSE loss function penalizes larger errors more heavily")
    print("3. Gradient descent iteratively improves parameters using gradients")
    print("4. Different optimizers can affect training speed and final performance")
    print("5. R² score tells us how much variance our model explains")
    print("6. The model successfully learned parameters close to the true values")
    print()
    
    # Optional: Plot training history if matplotlib is available
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Training loss over time
        plt.subplot(1, 2, 1)
        epochs = [h['epoch'] for h in training_history['history']]
        losses = [h['loss'] for h in training_history['history']]
        plt.plot(epochs, losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training Loss Over Time')
        plt.grid(True)
        
        # Plot 2: Predictions vs True values
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_test_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'Predictions vs True Values (R² = {test_r2:.3f})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('linear_regression_results.png', dpi=150, bbox_inches='tight')
        print("📊 Plots saved as 'linear_regression_results.png'")
        
    except ImportError:
        print("📊 Install matplotlib to see training plots")
    
    print()
    print("="*60)
    print("EXAMPLE COMPLETED SUCCESSFULLY! 🎉")
    print("="*60)


if __name__ == "__main__":
    main()
