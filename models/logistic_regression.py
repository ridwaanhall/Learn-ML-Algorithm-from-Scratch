"""
Logistic Regression implementation from scratch
"""
import numpy as np
from .base_model import BaseClassifier
from ..utils.matrix import MatrixUtils
from ..loss_functions.cross_entropy import CrossEntropyLoss
from ..optimization.sgd import SGDOptimizer
from ..optimization.adam import AdamOptimizer
from typing import Optional, Union


class LogisticRegression(BaseClassifier):
    """
    Logistic Regression classifier
    
    Uses logistic function (sigmoid) to model probability of binary classification
    For multiclass, uses one-vs-rest strategy
    """
    
    def __init__(self, penalty: Optional[str] = None, C: float = 1.0, 
                 fit_intercept: bool = True, solver: str = 'sgd', 
                 max_iter: int = 1000, learning_rate: float = 0.01, 
                 tolerance: float = 1e-6, multi_class: str = 'ovr', 
                 random_state: Optional[int] = None):
        """
        Initialize Logistic Regression
        
        Args:
            penalty: Regularization type ('l2', None)
            C: Inverse of regularization strength
            fit_intercept: Whether to fit intercept
            solver: Optimization algorithm ('sgd', 'adam')
            max_iter: Maximum number of iterations
            learning_rate: Learning rate for optimization
            tolerance: Tolerance for stopping criterion
            multi_class: Multiclass strategy ('ovr' for one-vs-rest)
            random_state: Random seed
        """
        super().__init__()
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.multi_class = multi_class
        self.random_state = random_state
        
        # Model parameters
        self.weights = None
        self.intercept = None
        self.loss_history = []
        self.binary_classifiers = {}  # For multiclass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit logistic regression model
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        self._validate_input(X, y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Get unique classes
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        if self.n_classes == 2:
            # Binary classification
            self._fit_binary(X, y)
        else:
            # Multiclass classification using one-vs-rest
            self._fit_multiclass(X, y)
        
        self.is_fitted = True
        return self
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit binary logistic regression
        
        Args:
            X: Feature matrix
            y: Binary target vector
        """
        # Convert labels to 0/1
        y_binary = (y == self.classes_[1]).astype(int)
        
        # Add bias column if fitting intercept
        if self.fit_intercept:
            X_with_bias = MatrixUtils.add_bias_column(X)
        else:
            X_with_bias = X.copy()
        
        # Initialize weights
        n_features = X_with_bias.shape[1]
        weights = np.random.normal(0, 0.01, n_features)
        
        # Choose optimizer
        if self.solver == 'sgd':
            optimizer = SGDOptimizer(learning_rate=self.learning_rate)
        elif self.solver == 'adam':
            optimizer = AdamOptimizer(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        self.loss_history = []
        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # Forward pass
            z = MatrixUtils.dot_product(X_with_bias, weights)
            y_pred_proba = MatrixUtils.sigmoid(z)
            
            # Compute loss
            loss = CrossEntropyLoss.binary_cross_entropy(y_binary, y_pred_proba)
            
            # Add regularization
            if self.penalty == 'l2':
                l2_penalty = np.sum(weights ** 2) / (2 * self.C)
                loss += l2_penalty
            
            self.loss_history.append(loss)
            
            # Check convergence
            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss
            
            # Compute gradients
            residuals = y_pred_proba - y_binary
            gradients = MatrixUtils.dot_product(MatrixUtils.transpose(X_with_bias), residuals) / len(y_binary)
            
            # Add regularization gradient
            if self.penalty == 'l2':
                reg_grad = weights / self.C
                if self.fit_intercept:
                    reg_grad[0] = 0  # Don't regularize intercept
                gradients += reg_grad
            
            # Update weights
            weights = optimizer.update(weights, gradients)
        
        # Store weights and intercept
        if self.fit_intercept:
            self.intercept = weights[0]
            self.weights = weights[1:]
        else:
            self.intercept = 0
            self.weights = weights
    
    def _fit_multiclass(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit multiclass logistic regression using one-vs-rest
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        self.binary_classifiers = {}
        
        for class_label in self.classes_:
            # Create binary target: current class vs all others
            y_binary = (y == class_label).astype(int)
            
            # Create and fit binary classifier
            binary_clf = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                fit_intercept=self.fit_intercept,
                solver=self.solver,
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                tolerance=self.tolerance,
                random_state=self.random_state
            )
            
            binary_clf._fit_binary(X, np.array([0, 1]))  # Dummy classes for binary fit
            binary_clf.classes_ = np.array([0, 1])
            binary_clf.n_classes = 2
            
            # Manually set the binary target and refit
            if self.fit_intercept:
                X_with_bias = MatrixUtils.add_bias_column(X)
            else:
                X_with_bias = X.copy()
            
            # Initialize weights
            n_features = X_with_bias.shape[1]
            weights = np.random.normal(0, 0.01, n_features)
            
            # Choose optimizer
            if self.solver == 'sgd':
                optimizer = SGDOptimizer(learning_rate=self.learning_rate)
            elif self.solver == 'adam':
                optimizer = AdamOptimizer(learning_rate=self.learning_rate)
            else:
                raise ValueError(f"Unknown solver: {self.solver}")
            
            prev_loss = float('inf')
            
            for iteration in range(self.max_iter):
                # Forward pass
                z = MatrixUtils.dot_product(X_with_bias, weights)
                y_pred_proba = MatrixUtils.sigmoid(z)
                
                # Compute loss
                loss = CrossEntropyLoss.binary_cross_entropy(y_binary, y_pred_proba)
                
                # Add regularization
                if self.penalty == 'l2':
                    l2_penalty = np.sum(weights ** 2) / (2 * self.C)
                    loss += l2_penalty
                
                # Check convergence
                if abs(prev_loss - loss) < self.tolerance:
                    break
                prev_loss = loss
                
                # Compute gradients
                residuals = y_pred_proba - y_binary
                gradients = MatrixUtils.dot_product(MatrixUtils.transpose(X_with_bias), residuals) / len(y_binary)
                
                # Add regularization gradient
                if self.penalty == 'l2':
                    reg_grad = weights / self.C
                    if self.fit_intercept:
                        reg_grad[0] = 0  # Don't regularize intercept
                    gradients += reg_grad
                
                # Update weights
                weights = optimizer.update(weights, gradients)
            
            # Store weights and intercept for this class
            if self.fit_intercept:
                binary_clf.intercept = weights[0]
                binary_clf.weights = weights[1:]
            else:
                binary_clf.intercept = 0
                binary_clf.weights = weights
            
            binary_clf.is_fitted = True
            self.binary_classifiers[class_label] = binary_clf
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        self._check_fitted()
        self._validate_input(X)
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features, but model was fitted with {self.n_features}")
        
        if self.n_classes == 2:
            # Binary classification
            z = MatrixUtils.dot_product(X, self.weights)
            if self.fit_intercept:
                z += self.intercept
            
            proba_class_1 = MatrixUtils.sigmoid(z)
            proba_class_0 = 1 - proba_class_1
            
            return np.column_stack([proba_class_0, proba_class_1])
        else:
            # Multiclass: one-vs-rest
            n_samples = X.shape[0]
            probabilities = np.zeros((n_samples, self.n_classes))
            
            for i, class_label in enumerate(self.classes_):
                binary_clf = self.binary_classifiers[class_label]
                z = MatrixUtils.dot_product(X, binary_clf.weights)
                if binary_clf.fit_intercept:
                    z += binary_clf.intercept
                
                probabilities[:, i] = MatrixUtils.sigmoid(z)
            
            # Normalize probabilities
            prob_sums = np.sum(probabilities, axis=1, keepdims=True)
            prob_sums[prob_sums == 0] = 1  # Avoid division by zero
            probabilities = probabilities / prob_sums
            
            return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes_[predicted_indices]
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Predict confidence scores for samples
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Confidence scores
        """
        self._check_fitted()
        self._validate_input(X)
        
        if self.n_classes == 2:
            z = MatrixUtils.dot_product(X, self.weights)
            if self.fit_intercept:
                z += self.intercept
            return z
        else:
            # For multiclass, return scores for all classes
            n_samples = X.shape[0]
            scores = np.zeros((n_samples, self.n_classes))
            
            for i, class_label in enumerate(self.classes_):
                binary_clf = self.binary_classifiers[class_label]
                z = MatrixUtils.dot_product(X, binary_clf.weights)
                if binary_clf.fit_intercept:
                    z += binary_clf.intercept
                scores[:, i] = z
            
            return scores
