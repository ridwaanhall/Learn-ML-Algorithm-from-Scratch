# Learn Machine Learning Algorithms from Scratch

A comprehensive educational project for understanding machine learning algorithms by implementing them from scratch using only NumPy. Perfect for beginners who want to understand the mathematical foundations and inner workings of ML algorithms.

**Created by [Ridwan Hall](https://ridwaanhall.com) for educational purposes.**

## 🎯 Project Goals

- **Educational Focus**: Learn how ML algorithms work under the hood
- **Mathematical Understanding**: Implement algorithms using mathematical foundations
- **No Black Boxes**: Build everything from scratch (except NumPy for mathematical operations)
- **Best Practices**: Learn when and how to use different algorithms, optimizers, and loss functions
- **Practical Knowledge**: Understand combinations and use cases for different components

## 📁 Project Structure

```
Learn-ML-Algorithm-from-Scratch/
├── src/
│   ├── models/              # ML algorithms implementation
│   │   ├── linear_regression.py
│   │   ├── logistic_regression.py
│   │   ├── neural_network.py
│   │   ├── decision_tree.py
│   │   ├── kmeans.py
│   │   └── __init__.py
│   ├── optimization/        # Optimization algorithms
│   │   ├── gradient_descent.py
│   │   ├── adam.py
│   │   ├── rmsprop.py
│   │   ├── momentum.py
│   │   └── __init__.py
│   ├── loss_functions/      # Loss functions
│   │   ├── mse.py
│   │   ├── cross_entropy.py
│   │   ├── mae.py
│   │   ├── huber.py
│   │   └── __init__.py
│   ├── activation_functions/ # Activation functions
│   │   ├── sigmoid.py
│   │   ├── relu.py
│   │   ├── tanh.py
│   │   ├── softmax.py
│   │   └── __init__.py
│   ├── metrics/             # Evaluation metrics
│   │   ├── accuracy.py
│   │   ├── precision_recall.py
│   │   ├── f1_score.py
│   │   ├── r2_score.py
│   │   └── __init__.py
│   ├── preprocessing/       # Data preprocessing
│   │   ├── scaler.py
│   │   ├── encoder.py
│   │   ├── splitter.py
│   │   └── __init__.py
│   └── utils/              # Utility functions
│       ├── data_loader.py
│       ├── visualizer.py
│       └── __init__.py
├── examples/               # Example implementations
├── docs/                  # Detailed documentation
├── tests/                 # Unit tests
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd Learn-ML-Algorithm-from-Scratch
   pip install -r requirements.txt
   ```

2. **Run Your First Example**:
   ```python
   from src.models.linear_regression import LinearRegression
   from src.loss_functions.mse import MSE
   from src.optimization.gradient_descent import GradientDescent
   
   # Create model with components
   model = LinearRegression()
   loss_fn = MSE()
   optimizer = GradientDescent(learning_rate=0.01)
   
   # Train the model
   model.fit(X_train, y_train, loss_fn, optimizer, epochs=1000)
   ```

## 📚 Learning Path

### 1. **Start with Fundamentals**
- **Loss Functions** (`docs/loss_functions.md`)
- **Activation Functions** (`docs/activation_functions.md`)
- **Optimization Algorithms** (`docs/optimization.md`)

### 2. **Simple Models**
- **Linear Regression** - Perfect for understanding gradient descent
- **Logistic Regression** - Introduction to classification

### 3. **Advanced Models**
- **Neural Networks** - Combining all concepts
- **Decision Trees** - Different approach to ML
- **K-Means** - Unsupervised learning

## 🎓 Educational Features

### Interactive Learning
- Each component has detailed docstrings explaining the math
- Step-by-step implementation with comments
- Visual examples in Jupyter notebooks

### Best Practices Guide
- **When to use Adam vs SGD**: See `docs/optimizer_guide.md`
- **Loss function selection**: See `docs/loss_function_guide.md`
- **Activation function combinations**: See `docs/activation_guide.md`

### Mathematical Foundations
- All algorithms implemented with clear mathematical derivations
- Gradient calculations shown step-by-step
- No hidden complexity - you see everything

## 🔧 Key Components

### Models
- **Linear Regression**: Fundamental regression algorithm
- **Logistic Regression**: Binary and multiclass classification
- **Neural Network**: Fully connected networks with backpropagation
- **Decision Tree**: Tree-based learning algorithm
- **K-Means**: Clustering algorithm

### Optimizers
- **Gradient Descent**: Basic optimization
- **Adam**: Adaptive learning rate with momentum
- **RMSprop**: Adaptive learning rate
- **Momentum**: Enhanced gradient descent

### Loss Functions
- **MSE**: Mean Squared Error for regression
- **Cross Entropy**: For classification tasks
- **MAE**: Mean Absolute Error for robust regression
- **Huber**: Robust loss function

### Activation Functions
- **ReLU**: Most popular for hidden layers
- **Sigmoid**: For binary classification output
- **Tanh**: Alternative to sigmoid
- **Softmax**: For multiclass classification

## 📖 Documentation

### 🚀 **Start Here** → [Project Overview](docs/project_overview.md)
**Complete guide to learning machine learning from scratch with structured paths and clear progression**

### Getting Started
- **[Quick Start Guide](docs/quickstart.md)** - Get up and running in 5 minutes
- **[Complete Learning Guide](docs/complete_learning_guide.md)** - Structured 8-week learning path
- **[Best Practices Guide](docs/best_practices.md)** - Choose the right components

### Technical References  
- **[API Reference](docs/api_reference.md)** - Complete documentation of all classes and methods
- **[Mathematical Foundations](docs/mathematical_foundations.md)** - Deep dive into the math behind each algorithm
- **[Troubleshooting Guide](docs/troubleshooting.md)** - Debug training issues and fix common problems

### Educational Flow
1. **Start Here**: `docs/project_overview.md` - Understand the complete learning journey
2. **Quick Start**: `docs/quickstart.md` - Run your first model
3. **Understand**: `docs/mathematical_foundations.md` - Learn the math  
4. **Master**: `docs/complete_learning_guide.md` - Follow the structured path
5. **Apply**: `docs/best_practices.md` - Make good choices
6. **Debug**: `docs/troubleshooting.md` - Fix problems
7. **Reference**: `docs/api_reference.md` - Look up details

## 🎯 Learning Outcomes

After completing this project, you will understand:

1. **How gradient descent actually works**
2. **Why certain optimizers work better for specific problems**
3. **How to choose appropriate loss functions**
4. **The mathematical foundations of neural networks**
5. **When to use different activation functions**
6. **How to implement ML algorithms from mathematical formulas**

## 👨‍💻 Author

**Ridwan Hall**  
Website: [ridwaanhall.com](https://ridwaanhall.com)

## 🤝 Contributing

This is an educational project! Contributions that enhance learning are welcome:
- Better documentation and explanations
- More visual examples
- Additional algorithms
- Improved mathematical explanations

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MIT License - Feel free to use this for educational purposes.

## 🎉 Get Started

Check out the `examples/` directory for hands-on tutorials, or dive into the documentation in `docs/` to understand the theory behind each component.

Happy Learning! 🚀