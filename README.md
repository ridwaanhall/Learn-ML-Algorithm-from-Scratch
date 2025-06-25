# 🤖 Machine Learning from Scratch

A comprehensive Python project implementing core Machine Learning algorithms from scratch using Object-Oriented Programming principles. This educational framework helps you understand how ML algorithms work internally without relying on high-level libraries.

## 🎯 Project Overview

This project implements a complete ML framework covering:
- **Supervised Learning**: Regression and Classification
- **Unsupervised Learning**: Clustering and Dimensionality Reduction
- **Optimization**: SGD and Adam optimizers
- **Preprocessing**: Data scaling, encoding, and splitting
- **Evaluation**: Comprehensive metrics for model assessment

## 📁 Project Structure

```
ml_from_scratch/
├── main.py                       # Entry point with comprehensive demos
├── data/
│   ├── iris.csv                  # Famous iris classification dataset
│   └── housing.csv               # Housing price regression dataset
├── models/
│   ├── __init__.py
│   ├── base_model.py            # Abstract base classes
│   ├── linear_regression.py      # Linear & Ridge Regression
│   ├── logistic_regression.py    # Logistic Regression
│   ├── decision_tree.py          # Decision Trees (CART)
│   ├── knn.py                    # K-Nearest Neighbors
│   ├── kmeans.py                 # K-Means Clustering
│   └── pca.py                    # Principal Component Analysis
├── optimization/
│   ├── __init__.py
│   ├── sgd.py                    # Stochastic Gradient Descent
│   └── adam.py                   # Adam Optimizer
├── loss_functions/
│   ├── __init__.py
│   ├── mse.py                    # Mean Squared Error
│   ├── mae.py                    # Mean Absolute Error
│   └── cross_entropy.py         # Cross Entropy Loss
├── metrics/
│   ├── __init__.py
│   ├── classification.py        # Classification metrics
│   └── regression.py            # Regression metrics
├── preprocessing/
│   ├── __init__.py
│   ├── scaler.py                # Data scaling utilities
│   ├── encoder.py               # Categorical encoding
│   └── split.py                 # Data splitting utilities
├── utils/
│   ├── __init__.py
│   └── matrix.py                # Matrix operations & utilities
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Learn-ML-Algorithm-from-Scratch
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Run the Demo

```bash
python main.py
```

This will run comprehensive demonstrations of all implemented algorithms.

## 🔧 Core Components

### 🎯 Supervised Learning

#### Linear Regression
```python
from models.linear_regression import LinearRegression

# Normal equation solver
model = LinearRegression(solver='normal_equation')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# SGD solver with custom parameters
model = LinearRegression(solver='sgd', learning_rate=0.01, max_iterations=1000)
```

#### Logistic Regression
```python
from models.logistic_regression import LogisticRegression

# Binary classification
model = LogisticRegression(max_iter=1000, learning_rate=0.1)
model.fit(X_train, y_train)

# Get probabilities
probabilities = model.predict_proba(X_test)
```

#### K-Nearest Neighbors
```python
from models.knn import KNNClassifier, KNNRegressor

# Classification
knn_clf = KNNClassifier(n_neighbors=5, weights='distance')
knn_clf.fit(X_train, y_train)

# Regression
knn_reg = KNNRegressor(n_neighbors=3, metric='euclidean')
knn_reg.fit(X_train, y_train)
```

#### Decision Trees
```python
from models.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

# Classification with Gini criterion
dt_clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
dt_clf.fit(X_train, y_train)

# Regression with squared error
dt_reg = DecisionTreeRegressor(criterion='squared_error', max_depth=8)
dt_reg.fit(X_train, y_train)
```

### 🔍 Unsupervised Learning

#### K-Means Clustering
```python
from models.kmeans import KMeans

# Basic clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# Access cluster centers
centroids = kmeans.cluster_centers_
```

#### Principal Component Analysis
```python
from models.pca import PCA

# Keep 95% of variance
pca = PCA(n_components=0.95)
X_transformed = pca.fit_transform(X)

# 2D visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X)
```

### ⚙️ Preprocessing

#### Data Scaling
```python
from preprocessing.scaler import StandardScaler, MinMaxScaler

# Standardization (zero mean, unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max scaling (0-1 range)
minmax = MinMaxScaler(feature_range=(0, 1))
X_minmax = minmax.fit_transform(X)
```

#### Categorical Encoding
```python
from preprocessing.encoder import LabelEncoder, OneHotEncoder

# Label encoding for ordinal data
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)

# One-hot encoding for nominal data
onehot_enc = OneHotEncoder()
X_encoded = onehot_enc.fit_transform(X_categorical)
```

#### Train-Test Split
```python
from preprocessing.split import TrainTestSplit

# Basic split
X_train, X_test, y_train, y_test = TrainTestSplit.train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Stratified split (maintains class distribution)
X_train, X_test, y_train, y_test = TrainTestSplit.train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
```

### 📊 Evaluation Metrics

#### Classification Metrics
```python
from metrics.classification import ClassificationMetrics

# Basic metrics
accuracy = ClassificationMetrics.accuracy(y_true, y_pred)
precision = ClassificationMetrics.precision(y_true, y_pred)
recall = ClassificationMetrics.recall(y_true, y_pred)
f1 = ClassificationMetrics.f1_score(y_true, y_pred)

# Comprehensive report
report = ClassificationMetrics.classification_report(y_true, y_pred)
```

#### Regression Metrics
```python
from metrics.regression import RegressionMetrics

# Basic metrics
r2 = RegressionMetrics.r2_score(y_true, y_pred)
mse = RegressionMetrics.mean_squared_error(y_true, y_pred)
rmse = RegressionMetrics.root_mean_squared_error(y_true, y_pred)
mae = RegressionMetrics.mean_absolute_error(y_true, y_pred)

# Comprehensive report
report = RegressionMetrics.regression_report(y_true, y_pred, n_features=X.shape[1])
```

### 🎛️ Optimization

#### SGD Optimizer
```python
from optimization.sgd import SGDOptimizer

optimizer = SGDOptimizer(learning_rate=0.01, momentum=0.9)
updated_params = optimizer.update(params, gradients)
```

#### Adam Optimizer
```python
from optimization.adam import AdamOptimizer

optimizer = AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
updated_params = optimizer.update(params, gradients)
```

## 📈 Algorithms Implemented

### Supervised Learning
- **Linear Regression**: Normal equation and SGD solvers
- **Ridge Regression**: L2 regularization
- **Logistic Regression**: Binary and multiclass (one-vs-rest)
- **K-Nearest Neighbors**: Classification and regression variants
- **Decision Trees**: CART algorithm with Gini/Entropy criteria

### Unsupervised Learning
- **K-Means Clustering**: Lloyd's algorithm with k-means++ initialization
- **PCA**: Eigendecomposition-based dimensionality reduction

### Optimization
- **Stochastic Gradient Descent**: With momentum support
- **Adam**: Adaptive moment estimation optimizer

### Loss Functions
- **Mean Squared Error**: For regression tasks
- **Mean Absolute Error**: Robust regression loss
- **Cross Entropy**: For classification tasks

## 🎓 Educational Features

### Object-Oriented Design
- **Abstract Base Classes**: Consistent API across all models
- **Inheritance**: Specialized classes for different model types
- **Encapsulation**: Clean separation of concerns
- **Polymorphism**: Unified interface for different algorithms

### Documentation
- **Comprehensive Docstrings**: Every class and method documented
- **Type Hints**: Clear parameter and return types
- **Comments**: Explaining algorithmic details and mathematical concepts

### Code Quality
- **Modular Architecture**: Organized into logical packages
- **Error Handling**: Robust input validation
- **Consistent API**: Follows scikit-learn conventions
- **Clean Code**: Readable and maintainable implementation

## 🔍 Example Usage

Here's a complete example using multiple components:

```python
import numpy as np
from models.logistic_regression import LogisticRegression
from preprocessing.scaler import StandardScaler
from preprocessing.split import TrainTestSplit
from metrics.classification import ClassificationMetrics

# Load your data
X, y = load_your_data()

# Split the data
X_train, X_test, y_train, y_test = TrainTestSplit.train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(learning_rate=0.1, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# Evaluate
accuracy = ClassificationMetrics.accuracy(y_test, y_pred)
report = ClassificationMetrics.classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:", report)
```

## 🎯 Learning Objectives

By exploring this codebase, you'll learn:

1. **Algorithm Implementation**: How ML algorithms work at a fundamental level
2. **Object-Oriented Design**: Best practices for structuring ML code
3. **Mathematical Foundations**: The math behind popular ML algorithms
4. **Software Engineering**: Clean, maintainable, and extensible code design
5. **Performance Considerations**: Numerical stability and computational efficiency

## 🔧 Dependencies

- **NumPy**: For numerical computations and linear algebra
- **Pandas**: For data manipulation and CSV loading
- **Matplotlib**: For visualization in demonstrations

See `requirements.txt` for specific versions.

## 🤝 Contributing

This is an educational project! Contributions are welcome:

1. **Algorithm Implementations**: Add new ML algorithms
2. **Optimizations**: Improve performance and numerical stability
3. **Documentation**: Enhance explanations and examples
4. **Testing**: Add unit tests and validation
5. **Features**: New preprocessing utilities or metrics

## 📚 Further Learning

### Books
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "Machine Learning: A Probabilistic Perspective" by Kevin Murphy

### Online Resources
- [scikit-learn documentation](https://scikit-learn.org/) for API reference
- [Coursera ML Course](https://www.coursera.org/learn/machine-learning) by Andrew Ng
- [CS229 Stanford](http://cs229.stanford.edu/) for mathematical foundations

## 📄 License

This project is intended for educational purposes. Feel free to use, modify, and learn from the code.

## 🎉 Acknowledgments

This implementation was inspired by:
- The scikit-learn API design
- Various educational ML resources and textbooks
- The need for understanding algorithms from first principles

---

**Happy Learning! 🚀**

Built with ❤️ for machine learning education.
