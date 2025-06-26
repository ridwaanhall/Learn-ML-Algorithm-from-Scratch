# API Reference Guide

Complete reference for all classes and functions in the Learn-ML-Algorithm-from-Scratch project.

## 📚 Overview

This project implements machine learning algorithms from scratch using only NumPy. All components follow consistent interfaces and are designed for educational purposes with extensive documentation.

## 🏗️ Architecture

### Core Design Principles
- **Modularity**: Each component is independent and reusable
- **Consistency**: Similar interfaces across all components
- **Educational**: Clear code with extensive comments and docstrings
- **Extensibility**: Easy to add new algorithms and modify existing ones

### Component Interaction
```
Data → Preprocessing → Model → Loss Function → Optimizer → Updated Model
```

## 🤖 Models (`src.models`)

### LinearRegression

Implements linear regression using gradient-based optimization.

```python
from src.models.linear_regression import LinearRegression

class LinearRegression:
    def __init__(self):
        """Initialize linear regression model."""
        
    def fit(self, X, y, loss_function, optimizer, epochs=1000, verbose=True):
        """
        Train the linear regression model.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features
        y : np.ndarray, shape (n_samples,)
            Training targets
        loss_function : Loss function object
            Must have forward() and backward() methods
        optimizer : Optimizer object
            Must have update() method
        epochs : int, default=1000
            Number of training iterations
        verbose : bool, default=True
            Whether to print training progress
            
        Returns:
        --------
        dict : Training history containing losses
        """
        
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        np.ndarray : Predicted values
        """
        
    def get_parameters(self):
        """Get current model parameters."""
        
    def set_parameters(self, weights, bias):
        """Set model parameters."""
```

**Example Usage:**
```python
model = LinearRegression()
history = model.fit(X_train, y_train, loss_fn, optimizer, epochs=1000)
predictions = model.predict(X_test)
```

## 📉 Loss Functions (`src.loss_functions`)

### MSE (Mean Squared Error)

```python
from src.loss_functions.mse import MSE

class MSE:
    def __init__(self):
        """Initialize MSE loss function."""
        
    def forward(self, y_true, y_pred):
        """
        Calculate MSE loss.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True target values
        y_pred : np.ndarray
            Predicted values
            
        Returns:
        --------
        float : MSE loss value
        """
        
    def backward(self, y_true, y_pred):
        """
        Calculate gradient of MSE loss.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True target values
        y_pred : np.ndarray
            Predicted values
            
        Returns:
        --------
        np.ndarray : Gradients with respect to predictions
        """
```

### CrossEntropy

```python
from src.loss_functions.cross_entropy import CrossEntropy

class CrossEntropy:
    def __init__(self, binary=True):
        """
        Initialize cross entropy loss.
        
        Parameters:
        -----------
        binary : bool, default=True
            Whether this is binary or multiclass classification
        """
        
    def forward(self, y_true, y_pred):
        """Calculate cross entropy loss."""
        
    def backward(self, y_true, y_pred):
        """Calculate gradients."""
```

### MAE (Mean Absolute Error)

```python
from src.loss_functions.mae import MAE

class MAE:
    def forward(self, y_true, y_pred):
        """Calculate MAE loss."""
        
    def backward(self, y_true, y_pred):
        """Calculate gradients (subgradient for non-differentiable points)."""
```

### Huber Loss

```python
from src.loss_functions.huber import HuberLoss

class HuberLoss:
    def __init__(self, delta=1.0):
        """
        Initialize Huber loss.
        
        Parameters:
        -----------
        delta : float, default=1.0
            Threshold for switching between quadratic and linear loss
        """
```

## ⚡ Optimizers (`src.optimization`)

### GradientDescent

```python
from src.optimization.gradient_descent import GradientDescent

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        """
        Initialize gradient descent optimizer.
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Step size for parameter updates
        """
        
    def update(self, parameters, gradients):
        """
        Update parameters using gradients.
        
        Parameters:
        -----------
        parameters : dict
            Current parameter values
        gradients : dict
            Gradients for each parameter
            
        Returns:
        --------
        dict : Updated parameters
        """
```

### Adam

```python
from src.optimization.adam import Adam

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer.
        
        Parameters:
        -----------
        learning_rate : float, default=0.001
            Step size for parameter updates
        beta1 : float, default=0.9
            Exponential decay rate for first moment estimates
        beta2 : float, default=0.999
            Exponential decay rate for second moment estimates
        epsilon : float, default=1e-8
            Small constant for numerical stability
        """
```

### RMSprop

```python
from src.optimization.rmsprop import RMSprop

class RMSprop:
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        """Initialize RMSprop optimizer."""
```

### Momentum

```python
from src.optimization.momentum import Momentum, NesterovMomentum

class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """Initialize momentum optimizer."""

class NesterovMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """Initialize Nesterov momentum optimizer."""
```

## 🧠 Activation Functions (`src.activation_functions`)

### ReLU

```python
from src.activation_functions.relu import ReLU

class ReLU:
    def __init__(self):
        """Initialize ReLU activation function."""
        
    def forward(self, x):
        """
        Apply ReLU activation.
        
        Parameters:
        -----------
        x : np.ndarray
            Input values
            
        Returns:
        --------
        np.ndarray : max(0, x)
        """
        
    def backward(self, grad_output):
        """
        Calculate gradients for backpropagation.
        
        Parameters:
        -----------
        grad_output : np.ndarray
            Gradients from the next layer
            
        Returns:
        --------
        np.ndarray : Gradients with respect to input
        """
```

### Sigmoid

```python
from src.activation_functions.sigmoid import Sigmoid

class Sigmoid:
    def forward(self, x):
        """Apply sigmoid activation: 1 / (1 + exp(-x))"""
        
    def backward(self, grad_output):
        """Calculate gradients: sigmoid(x) * (1 - sigmoid(x))"""
```

### Tanh

```python
from src.activation_functions.tanh import Tanh

class Tanh:
    def forward(self, x):
        """Apply tanh activation."""
        
    def backward(self, grad_output):
        """Calculate gradients: 1 - tanh²(x)"""
```

### Softmax

```python
from src.activation_functions.softmax import Softmax

class Softmax:
    def forward(self, x):
        """
        Apply softmax activation.
        
        Returns probability distribution over classes.
        """
```

## 📊 Metrics (`src.metrics`)

### Accuracy

```python
from src.metrics.accuracy import Accuracy

class Accuracy:
    def calculate(self, y_true, y_pred):
        """
        Calculate classification accuracy.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
            
        Returns:
        --------
        float : Accuracy score between 0 and 1
        """
```

### Precision and Recall

```python
from src.metrics.precision_recall import Precision, Recall

class Precision:
    def __init__(self, average='binary'):
        """
        Initialize precision metric.
        
        Parameters:
        -----------
        average : str, default='binary'
            Averaging strategy: 'binary', 'macro', 'micro'
        """

class Recall:
    def __init__(self, average='binary'):
        """Initialize recall metric."""
```

### F1Score

```python
from src.metrics.f1_score import F1Score

class F1Score:
    def calculate(self, y_true, y_pred):
        """Calculate F1 score (harmonic mean of precision and recall)."""
        
    def calculate_with_components(self, y_true, y_pred):
        """
        Calculate F1 score along with precision and recall.
        
        Returns:
        --------
        dict : {'f1_score': float, 'precision': float, 'recall': float}
        """
```

### R2Score

```python
from src.metrics.r2_score import R2Score

class R2Score:
    def calculate(self, y_true, y_pred):
        """Calculate R-squared (coefficient of determination)."""
        
    def calculate_adjusted(self, y_true, y_pred, n_features):
        """Calculate adjusted R-squared."""
        
    def explained_variance_score(self, y_true, y_pred):
        """Calculate explained variance score."""
```

## 🔄 Preprocessing (`src.preprocessing`)

### Scalers

```python
from src.preprocessing.scaler import StandardScaler, MinMaxScaler

class StandardScaler:
    def fit(self, X):
        """Learn mean and standard deviation from training data."""
        
    def transform(self, X):
        """Apply standardization: (X - mean) / std"""
        
    def fit_transform(self, X):
        """Fit and transform in one step."""
        
    def inverse_transform(self, X):
        """Reverse the standardization."""

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        """
        Initialize Min-Max scaler.
        
        Parameters:
        -----------
        feature_range : tuple, default=(0, 1)
            Target range for scaling
        """
```

### Encoders

```python
from src.preprocessing.encoder import LabelEncoder, OneHotEncoder

class LabelEncoder:
    def fit(self, y):
        """Learn unique labels."""
        
    def transform(self, y):
        """Convert labels to integers."""
        
    def inverse_transform(self, y):
        """Convert integers back to original labels."""

class OneHotEncoder:
    def fit(self, X):
        """Learn unique categories for each feature."""
        
    def transform(self, X):
        """Convert to one-hot encoding."""
```

### Data Splitters

```python
from src.preprocessing.splitter import (
    TrainTestSplit, KFold, StratifiedSplit, train_test_split
)

class TrainTestSplit:
    def __init__(self, test_size=0.2, random_state=None, shuffle=True):
        """Initialize train-test splitter."""
        
    def split(self, X, y=None):
        """Split data into train and test sets."""

class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        """Initialize K-Fold cross-validation."""
        
    def split(self, X, y=None):
        """Generate K-Fold splits."""

def train_test_split(X, y=None, test_size=0.2, random_state=None, shuffle=True):
    """Convenience function for train-test splitting."""
```

## 🛠️ Utilities (`src.utils`)

### DataLoader

```python
from src.utils.data_loader import DataLoader

class DataLoader:
    def load_csv(self, filepath, target_column=None):
        """Load data from CSV file."""
        
    def create_synthetic_regression(self, n_samples=100, n_features=1, noise=0.1):
        """Generate synthetic regression data."""
        
    def create_synthetic_classification(self, n_samples=100, n_features=2, n_classes=2):
        """Generate synthetic classification data."""
```

### ModelVisualizer

```python
from src.utils.visualizer import ModelVisualizer

class ModelVisualizer:
    def __init__(self, figsize=(10, 6)):
        """Initialize visualizer."""
        
    def plot_regression_results(self, y_true, y_pred, title="Regression Results"):
        """Plot actual vs predicted values and residuals."""
        
    def plot_training_history(self, losses, title="Training History"):
        """Plot training loss over epochs."""
        
    def plot_classification_results(self, y_true, y_pred, class_names=None):
        """Plot confusion matrix for classification."""
        
    def plot_data_distribution(self, X, y=None, feature_names=None):
        """Plot histograms of feature distributions."""
        
    def plot_learning_curve(self, train_losses, val_losses=None):
        """Plot training and validation losses."""
        
    def plot_decision_boundary_2d(self, X, y, model, title="Decision Boundary"):
        """Plot decision boundary for 2D classification problems."""
```

## 🔧 Common Usage Patterns

### Basic Training Loop
```python
# Initialize components
model = LinearRegression()
loss_fn = MSE()
optimizer = GradientDescent(learning_rate=0.01)

# Train the model
history = model.fit(X_train, y_train, loss_fn, optimizer, epochs=1000)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
r2 = R2Score().calculate(y_test, predictions)
```

### Data Preprocessing Pipeline
```python
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

### Comparison of Optimizers
```python
optimizers = {
    'SGD': GradientDescent(learning_rate=0.01),
    'Adam': Adam(learning_rate=0.001),
    'RMSprop': RMSprop(learning_rate=0.001)
}

for name, opt in optimizers.items():
    model = LinearRegression()
    history = model.fit(X, y, MSE(), opt, epochs=1000)
    print(f"{name}: Final loss = {history['losses'][-1]:.4f}")
```

## 🐛 Error Handling

### Common Exceptions
- **ValueError**: Invalid input shapes or parameters
- **TypeError**: Incorrect data types
- **RuntimeError**: Convergence issues or numerical instability

### Debugging Tips
```python
# Check for NaN values
assert not np.isnan(X).any(), "Input contains NaN values"

# Check array shapes
assert X.shape[0] == y.shape[0], "X and y must have same number of samples"

# Monitor training
if np.isnan(loss):
    print("Loss is NaN - try reducing learning rate")
```

## 📋 Type Hints

All functions include proper type hints:

```python
def fit(self, 
        X: np.ndarray, 
        y: np.ndarray, 
        loss_function: Any, 
        optimizer: Any, 
        epochs: int = 1000) -> Dict[str, List[float]]:
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python tests/run_all_tests.py

# Run specific test modules
python tests/test_loss_functions.py
python tests/test_optimization.py
python tests/test_activation_functions.py
```

### Creating Custom Tests
```python
def test_custom_functionality():
    # Setup
    model = YourCustomModel()
    
    # Test
    result = model.some_method(test_input)
    
    # Assert
    assert result.shape == expected_shape
    assert np.allclose(result, expected_output)
```

## 🔗 Integration Examples

### Complete ML Pipeline
```python
from src.preprocessing.splitter import train_test_split
from src.preprocessing.scaler import StandardScaler
from src.models.linear_regression import LinearRegression
from src.loss_functions.mse import MSE
from src.optimization.adam import Adam
from src.metrics.r2_score import R2Score
from src.utils.visualizer import ModelVisualizer

# Load and preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
optimizer = Adam(learning_rate=0.001)
loss_fn = MSE()
history = model.fit(X_train_scaled, y_train, loss_fn, optimizer)

# Evaluate and visualize
predictions = model.predict(X_test_scaled)
r2 = R2Score().calculate(y_test, predictions)

visualizer = ModelVisualizer()
visualizer.plot_regression_results(y_test, predictions)
visualizer.plot_training_history(history['losses'])
```

This API reference provides complete documentation for using all components in the project. Each class and method includes detailed parameter descriptions, return values, and usage examples. 🚀
