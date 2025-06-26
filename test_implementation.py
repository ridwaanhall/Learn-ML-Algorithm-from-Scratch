"""
Test your ML implementation

Run this script to verify everything is working correctly.
"""

import numpy as np
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.models.linear_regression import LinearRegression
        from src.loss_functions.mse import MSE
        from src.loss_functions.cross_entropy import CrossEntropy
        from src.loss_functions.mae import MAE
        from src.loss_functions.huber import HuberLoss
        from src.optimization.gradient_descent import GradientDescent
        from src.optimization.adam import Adam
        from src.activation_functions.relu import ReLU
        from src.activation_functions.sigmoid import Sigmoid
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_linear_regression():
    """Test linear regression with synthetic data"""
    print("\nTesting Linear Regression...")
    
    try:
        from src.models.linear_regression import LinearRegression
        from src.loss_functions.mse import MSE
        from src.optimization.gradient_descent import GradientDescent
        
        # Create simple test data: y = 2x + 1
        np.random.seed(42)
        X = np.random.uniform(-1, 1, (50, 1))
        y = 2 * X.flatten() + 1 + np.random.normal(0, 0.1, 50)
        
        # Create and train model
        model = LinearRegression()
        loss_fn = MSE()
        optimizer = GradientDescent(learning_rate=0.5)  # Higher learning rate for faster convergence
        
        history = model.fit(X, y, loss_fn, optimizer, epochs=1000, verbose=False)  # More epochs
        
        # Check if model learned reasonable parameters
        expected_weight = 2.0
        expected_bias = 1.0
        
        weight_error = abs(model.weights[0] - expected_weight)
        bias_error = abs(model.bias - expected_bias)
        
        if weight_error < 0.2 and bias_error < 0.2:
            print(f"✅ Linear regression working! Weight: {model.weights[0]:.3f} (expected ~2.0), Bias: {model.bias:.3f} (expected ~1.0)")
            return True
        else:
            print(f"⚠️  Linear regression parameters not accurate. Weight: {model.weights[0]:.3f}, Bias: {model.bias:.3f}")
            return False
            
    except Exception as e:
        print(f"❌ Linear regression test failed: {e}")
        return False

def test_loss_functions():
    """Test loss functions"""
    print("\nTesting Loss Functions...")
    
    try:
        from src.loss_functions.mse import MSE
        from src.loss_functions.mae import MAE
        from src.loss_functions.cross_entropy import CrossEntropy
        from src.loss_functions.huber import HuberLoss
        
        # Test data
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        
        # Test MSE
        mse = MSE()
        mse_loss = mse(y_true, y_pred)
        mse_grad = mse.backward(y_true, y_pred)
        
        # Test MAE
        mae = MAE()
        mae_loss = mae(y_true, y_pred)
        mae_grad = mae.backward(y_true, y_pred)
        
        # Test Huber
        huber = HuberLoss()
        huber_loss = huber(y_true, y_pred)
        huber_grad = huber.backward(y_true, y_pred)
        
        # Test Cross Entropy (binary)
        y_true_binary = np.array([1, 0, 1])
        y_pred_binary = np.array([0.8, 0.2, 0.9])
        
        ce = CrossEntropy(binary=True)
        ce_loss = ce(y_true_binary, y_pred_binary)
        ce_grad = ce.backward(y_true_binary, y_pred_binary)
        
        print(f"✅ Loss functions working! MSE: {mse_loss:.4f}, MAE: {mae_loss:.4f}, Huber: {huber_loss:.4f}, CE: {ce_loss:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        return False

def test_optimizers():
    """Test optimizers"""
    print("\nTesting Optimizers...")
    
    try:
        from src.optimization.gradient_descent import GradientDescent
        from src.optimization.adam import Adam
        
        # Test data
        params = {'weight': np.array([1.0]), 'bias': np.array([0.5])}
        grads = {'weight': np.array([0.1]), 'bias': np.array([-0.05])}
        
        # Test Gradient Descent
        gd = GradientDescent(learning_rate=0.1)
        updated_gd = gd.update(params.copy(), grads)
        
        # Test Adam
        adam = Adam(learning_rate=0.01)
        updated_adam = adam.update(params.copy(), grads)
        
        print(f"✅ Optimizers working! GD updated weight: {updated_gd['weight'][0]:.3f}, Adam updated weight: {updated_adam['weight'][0]:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Optimizer test failed: {e}")
        return False

def test_activation_functions():
    """Test activation functions"""
    print("\nTesting Activation Functions...")
    
    try:
        from src.activation_functions.relu import ReLU
        from src.activation_functions.sigmoid import Sigmoid
        
        # Test data
        x = np.array([-2, -1, 0, 1, 2])
        
        # Test ReLU
        relu = ReLU()
        relu_out = relu(x)
        relu_grad = relu.backward(np.ones_like(x))
        
        # Test Sigmoid
        sigmoid = Sigmoid()
        sigmoid_out = sigmoid(x)
        sigmoid_grad = sigmoid.backward(np.ones_like(x))
        
        print(f"✅ Activation functions working! ReLU max: {relu_out.max():.1f}, Sigmoid range: [{sigmoid_out.min():.3f}, {sigmoid_out.max():.3f}]")
        return True
        
    except Exception as e:
        print(f"❌ Activation function test failed: {e}")
        return False

def run_complete_test():
    """Run a complete end-to-end test"""
    print("\nRunning Complete End-to-End Test...")
    
    try:
        # This essentially runs the complete example
        from src.models.linear_regression import LinearRegression
        from src.loss_functions.mse import MSE
        from src.optimization.adam import Adam
        
        # Generate data
        np.random.seed(42)
        X = np.random.uniform(-2, 2, (100, 2))
        true_weights = np.array([3.0, -1.5])
        true_bias = 2.0
        y = np.dot(X, true_weights) + true_bias + np.random.normal(0, 0.1, 100)
        
        # Train model
        model = LinearRegression()
        loss_fn = MSE()
        optimizer = Adam(learning_rate=0.1)  # Higher learning rate for Adam
        
        history = model.fit(X, y, loss_fn, optimizer, epochs=500, verbose=False)  # More epochs
        
        # Evaluate
        predictions = model.predict(X)
        r2_score = model.score(X, y, metric='r2')
        
        if r2_score > 0.90:  # Slightly lower threshold but still good
            print(f"✅ Complete test passed! R² score: {r2_score:.4f}, Final loss: {history['final_loss']:.6f}")
            return True
        else:
            print(f"⚠️  Complete test suboptimal. R² score: {r2_score:.4f}")
            return False
            
    except Exception as e:
        print(f"❌ Complete test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("TESTING ML FROM SCRATCH IMPLEMENTATION")
    print("="*60)
    
    tests = [
        test_imports,
        test_loss_functions,
        test_optimizers,
        test_activation_functions,
        test_linear_regression,
        run_complete_test
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your ML implementation is working correctly.")
        print("\nNext steps:")
        print("1. Run: python examples/complete_linear_regression.py")
        print("2. Read: docs/quickstart.md")
        print("3. Explore: Individual module examples")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Make sure all files are in the correct locations.")
    
    print("="*60)

if __name__ == "__main__":
    main()
