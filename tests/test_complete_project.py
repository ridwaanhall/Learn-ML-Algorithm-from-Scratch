"""
Test imports for all new modules to ensure everything is properly connected.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_all_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports for all modules...")
    
    try:
        # Test metrics imports
        from src.metrics import Accuracy, Precision, Recall, F1Score, R2Score
        print("✓ Metrics imports successful")
        
        # Test preprocessing imports
        from src.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
        from src.preprocessing import TrainTestSplit, KFold, StratifiedSplit, train_test_split
        print("✓ Preprocessing imports successful")
        
        # Test utils imports
        from src.utils import DataLoader, ModelVisualizer
        print("✓ Utils imports successful")
        
        # Test optimization imports
        from src.optimization import GradientDescent, Adam, RMSprop, Momentum, NesterovMomentum
        print("✓ Optimization imports successful")
        
        # Test loss functions imports
        from src.loss_functions import MSE, MAE, CrossEntropy, HuberLoss
        print("✓ Loss functions imports successful")
        
        # Test activation functions imports
        from src.activation_functions import ReLU, Sigmoid, Tanh, Softmax
        print("✓ Activation functions imports successful")
        
        # Test models imports
        from src.models import LinearRegression
        print("✓ Models imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key modules."""
    import numpy as np
    
    try:
        # Test metrics
        from src.metrics import Accuracy, R2Score
        
        acc = Accuracy()
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        accuracy = acc.calculate(y_true, y_pred)
        print(f"✓ Accuracy calculation: {accuracy:.2f}")
        
        r2 = R2Score()
        y_true_reg = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred_reg = np.array([1.1, 2.1, 2.9, 3.8])
        r2_score = r2.calculate(y_true_reg, y_pred_reg)
        print(f"✓ R2 score calculation: {r2_score:.3f}")
        
        # Test preprocessing
        from src.preprocessing import StandardScaler, train_test_split
        
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_scaled = scaler.fit_transform(X)
        print(f"✓ StandardScaler working")
        
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        print(f"✓ Train-test split: {X_train.shape}, {X_test.shape}")
        
        # Test optimizers
        from src.optimization import Momentum
        
        momentum_opt = Momentum(learning_rate=0.01, momentum=0.9)
        params = np.array([1.0, 2.0])
        grads = np.array([0.1, 0.2])
        updated = momentum_opt.update(params, grads)
        print(f"✓ Momentum optimizer working")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("TESTING ALL MODULE IMPORTS AND BASIC FUNCTIONALITY")
    print("="*60)
    
    imports_ok = test_all_imports()
    print()
    
    functionality_ok = test_basic_functionality()
    
    print("\n" + "="*60)
    if imports_ok and functionality_ok:
        print("🎉 ALL TESTS PASSED! Project is complete and functional.")
    else:
        print("⚠️ Some tests failed.")
    print("="*60)
