"""
Educational Example: Understanding Machine Learning Components

This example demonstrates how each component contributes to the overall learning process.
Perfect for beginners who want to understand what's happening at each step.

Author: Ridwan Hall (ridwaanhall.com)
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.linear_regression import LinearRegression
from src.loss_functions.mse import MSE
from src.loss_functions.mae import MAE
from src.optimization.gradient_descent import GradientDescent
from src.optimization.adam import Adam
from src.preprocessing.scaler import StandardScaler
from src.preprocessing.splitter import train_test_split
from src.metrics.r2_score import R2Score
from src.utils.visualizer import ModelVisualizer

def educational_demo():
    """
    Step-by-step demonstration of machine learning concepts.
    """
    print("="*80)
    print("EDUCATIONAL DEMO: Understanding Machine Learning from Scratch")
    print("="*80)
    
    # Step 1: Create and understand the data
    print("\n📊 STEP 1: Creating and Understanding Data")
    print("-" * 50)
    
    np.random.seed(42)
    n_samples = 100
    
    # Create synthetic data with known relationship
    X = np.random.uniform(-3, 3, (n_samples, 2))
    true_weights = np.array([2.5, -1.5])  # True relationship we want to discover
    true_bias = 1.0
    noise = np.random.normal(0, 0.3, n_samples)
    
    # True relationship: y = 2.5*x1 - 1.5*x2 + 1.0 + noise
    y = X @ true_weights + true_bias + noise
    
    print(f"✓ Generated {n_samples} samples with 2 features")
    print(f"✓ True relationship: y = {true_weights[0]}*x1 + {true_weights[1]}*x2 + {true_bias} + noise")
    print(f"✓ Data ranges: X1=[{X[:,0].min():.2f}, {X[:,0].max():.2f}], X2=[{X[:,1].min():.2f}, {X[:,1].max():.2f}]")
    print(f"✓ Target range: y=[{y.min():.2f}, {y.max():.2f}]")
    
    # Step 2: Preprocessing
    print("\n🔧 STEP 2: Data Preprocessing")
    print("-" * 50)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✓ Split data: {len(X_train)} training, {len(X_test)} test samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Before scaling - Feature means: {np.mean(X_train, axis=0)}")
    print(f"✓ After scaling - Feature means: {np.mean(X_train_scaled, axis=0)}")
    print(f"✓ Before scaling - Feature stds: {np.std(X_train, axis=0)}")
    print(f"✓ After scaling - Feature stds: {np.std(X_train_scaled, axis=0)}")
    
    # Step 3: Understanding different loss functions
    print("\n📉 STEP 3: Comparing Loss Functions")
    print("-" * 50)
    
    # Initialize models for comparison
    models = {}
    loss_functions = {'MSE': MSE(), 'MAE': MAE()}
    
    for loss_name, loss_fn in loss_functions.items():
        print(f"\n--- Training with {loss_name} Loss ---")
        
        model = LinearRegression()
        optimizer = GradientDescent(learning_rate=0.01)
        
        # Train the model
        history = model.fit(X_train_scaled, y_train, loss_fn, optimizer, epochs=1000, verbose=False)
        
        # Make predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Calculate performance
        r2_score = R2Score()
        train_r2 = r2_score.calculate(y_train, train_pred)
        test_r2 = r2_score.calculate(y_test, test_pred)
        
        models[loss_name] = {
            'model': model,
            'history': history,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_pred': test_pred
        }
        
        print(f"✓ Final loss: {history['final_loss']:.6f}")
        print(f"✓ Learned weights: {model.weights}")
        print(f"✓ Learned bias: {model.bias:.4f}")
        print(f"✓ Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"✓ True weights: {true_weights}, True bias: {true_bias}")
    
    # Step 4: Comparing optimizers
    print("\n⚡ STEP 4: Comparing Optimizers")
    print("-" * 50)
    
    optimizers = {
        'Gradient Descent': GradientDescent(learning_rate=0.01),
        'Adam': Adam(learning_rate=0.01)
    }
    
    optimizer_results = {}
    
    for opt_name, optimizer in optimizers.items():
        print(f"\n--- Training with {opt_name} ---")
        
        model = LinearRegression()
        loss_fn = MSE()
        
        # Train the model
        history = model.fit(X_train_scaled, y_train, loss_fn, optimizer, epochs=500, verbose=False)
        
        # Store results
        losses = [h['loss'] for h in history['history']]
        optimizer_results[opt_name] = {
            'history': history,
            'losses': losses,
            'final_loss': history['final_loss'],
            'convergence_epoch': next((i for i, loss in enumerate(losses) 
                                     if loss < 0.1), len(losses))
        }
        
        print(f"✓ Final loss: {history['final_loss']:.6f}")
        print(f"✓ Epochs to reach loss < 0.1: {optimizer_results[opt_name]['convergence_epoch']}")
    
    # Step 5: Visualization and Analysis
    print("\n📊 STEP 5: Visualization and Analysis")
    print("-" * 50)
    
    visualizer = ModelVisualizer()
    
    # Plot training histories for different optimizers
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Optimizer comparison
    plt.subplot(2, 3, 1)
    for opt_name, results in optimizer_results.items():
        plt.plot(results['losses'], label=opt_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Loss function comparison (final predictions)
    plt.subplot(2, 3, 2)
    for loss_name, results in models.items():
        plt.scatter(y_test, results['test_pred'], alpha=0.7, label=f'{loss_name} (R²={results["test_r2"]:.3f})')
    
    # Perfect prediction line
    min_val, max_val = y_test.min(), y_test.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Loss Function Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Feature importance visualization
    plt.subplot(2, 3, 3)
    best_model = models['MSE']['model']  # Use MSE model for analysis
    feature_names = ['Feature 1', 'Feature 2']
    
    plt.bar(feature_names, best_model.weights, alpha=0.7, color=['blue', 'orange'])
    plt.bar(feature_names, true_weights, alpha=0.5, color=['red', 'red'], label='True Weights')
    plt.ylabel('Weight Value')
    plt.title('Learned vs True Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Residual analysis
    plt.subplot(2, 3, 4)
    best_pred = models['MSE']['test_pred']
    residuals = y_test - best_pred
    plt.scatter(best_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis')
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Data distribution
    plt.subplot(2, 3, 5)
    plt.hist(y, bins=15, alpha=0.7, label='Target Distribution')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    plt.title('Target Distribution')
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Feature correlation
    plt.subplot(2, 3, 6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Target Value')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Feature Space Colored by Target')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('educational_demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 6: Key Insights
    print("\n💡 STEP 6: Key Insights and Takeaways")
    print("-" * 50)
    
    print("\n🔍 What We Learned:")
    print(f"✓ Data scaling is crucial - notice how feature means became ~0 and stds became ~1")
    print(f"✓ MSE vs MAE: Both converged to similar solutions for this clean data")
    print(f"✓ Adam vs SGD: Adam converged faster ({optimizer_results['Adam']['convergence_epoch']} vs {optimizer_results['Gradient Descent']['convergence_epoch']} epochs)")
    print(f"✓ Model recovered true weights reasonably well:")
    
    best_model = models['MSE']['model']
    weight_errors = np.abs(best_model.weights - true_weights)
    bias_error = abs(best_model.bias - true_bias)
    
    print(f"   - Weight errors: {weight_errors}")
    print(f"   - Bias error: {bias_error:.4f}")
    print(f"✓ R² score of {models['MSE']['test_r2']:.4f} indicates good fit")
    
    print("\n🎯 Next Steps for Learning:")
    print("1. Try adding more noise to see how it affects different loss functions")
    print("2. Experiment with different learning rates")
    print("3. Add polynomial features to handle non-linear relationships")
    print("4. Try different train/test splits to see generalization")
    print("5. Implement regularization to prevent overfitting")
    
    print("\n🎓 Educational Value:")
    print("- You now understand how each component contributes to learning")
    print("- You've seen how to diagnose model performance")
    print("- You can make informed choices about algorithms and hyperparameters")
    print("- You understand the importance of data preprocessing")
    
    return models, optimizer_results

def interactive_experiment():
    """
    Interactive experiment for students to modify and learn.
    """
    print("\n" + "="*80)
    print("INTERACTIVE EXPERIMENT: Try Your Own Modifications!")
    print("="*80)
    
    print("\n🧪 Experiment Ideas:")
    print("1. Change the noise level in data generation")
    print("2. Try different learning rates")
    print("3. Modify the true relationship")
    print("4. Add more features")
    print("5. Try different optimizers")
    
    # This would be extended with actual interactive components
    print("\n💻 Modify the code above and see how results change!")

if __name__ == "__main__":
    # Run the educational demo
    models, optimizer_results = educational_demo()
    
    # Run interactive experiment suggestions
    interactive_experiment()
    
    print("\n🎉 Demo Complete! Check the generated plot and try modifying the code.")
    print("This is how you truly learn machine learning - by experimenting! 🚀")
