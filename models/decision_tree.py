"""
Decision Tree implementation from scratch
"""
import numpy as np
from .base_model import BaseClassifier, BaseRegressor
from typing import Optional, Dict, Any, Union
from collections import Counter


class Node:
    """
    Node class for decision tree
    """
    
    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['Node'] = None, right: Optional['Node'] = None,
                 value: Optional[Union[int, float]] = None, samples: int = 0,
                 impurity: float = 0.0):
        """
        Initialize tree node
        
        Args:
            feature: Feature index for splitting
            threshold: Threshold value for splitting
            left: Left child node
            right: Right child node
            value: Prediction value (for leaf nodes)
            samples: Number of samples in this node
            impurity: Impurity measure at this node
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = samples
        self.impurity = impurity
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf"""
        return self.value is not None


class DecisionTreeClassifier(BaseClassifier):
    """
    Decision Tree Classifier using CART algorithm
    """
    
    def __init__(self, criterion: str = 'gini', max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: Optional[Union[int, str]] = None,
                 random_state: Optional[int] = None):
        """
        Initialize Decision Tree Classifier
        
        Args:
            criterion: Splitting criterion ('gini', 'entropy')
            max_depth: Maximum depth of tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            random_state: Random seed
        """
        super().__init__()
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
        self.root = None
        self.feature_importances_ = None
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """
        Calculate impurity based on criterion
        
        Args:
            y: Target values
            
        Returns:
            Impurity value
        """
        if len(y) == 0:
            return 0
        
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy"""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        # Avoid log(0)
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _information_gain(self, y: np.ndarray, left_mask: np.ndarray) -> float:
        """
        Calculate information gain from a split
        
        Args:
            y: Target values
            left_mask: Boolean mask for left split
            
        Returns:
            Information gain
        """
        if len(y) == 0:
            return 0
        
        # Parent impurity
        parent_impurity = self._calculate_impurity(y)
        
        # Children
        y_left = y[left_mask]
        y_right = y[~left_mask]
        
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        
        # Weighted average of children impurities
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        left_impurity = self._calculate_impurity(y_left)
        right_impurity = self._calculate_impurity(y_right)
        
        child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        
        # Information gain
        info_gain = parent_impurity - child_impurity
        return info_gain
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Find the best split for the data
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (best_feature, best_threshold, best_gain)
        """
        n_features = X.shape[1]
        
        # Determine features to consider
        if self.max_features is None:
            features_to_consider = range(n_features)
        elif isinstance(self.max_features, int):
            if self.random_state is not None:
                np.random.seed(self.random_state)
            features_to_consider = np.random.choice(n_features, 
                                                  min(self.max_features, n_features), 
                                                  replace=False)
        elif self.max_features == 'sqrt':
            n_features_to_consider = int(np.sqrt(n_features))
            if self.random_state is not None:
                np.random.seed(self.random_state)
            features_to_consider = np.random.choice(n_features, 
                                                  n_features_to_consider, 
                                                  replace=False)
        else:
            features_to_consider = range(n_features)
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in features_to_consider:
            # Get unique thresholds to try
            feature_values = X[:, feature]
            unique_values = np.unique(feature_values)
            
            # Try thresholds between unique values
            thresholds = []
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                thresholds.append(threshold)
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                
                # Check if split creates valid children
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(~left_mask) < self.min_samples_leaf:
                    continue
                
                gain = self._information_gain(y, left_mask)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively build the decision tree
        
        Args:
            X: Feature matrix
            y: Target vector
            depth: Current depth
            
        Returns:
            Root node of the tree
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Create node
        node = Node(samples=n_samples, impurity=self._calculate_impurity(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            # Create leaf node
            most_common_class = Counter(y).most_common(1)[0][0]
            node.value = most_common_class
            return node
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_feature is None or best_gain <= 0:
            # Create leaf node if no good split found
            most_common_class = Counter(y).most_common(1)[0][0]
            node.value = most_common_class
            return node
        
        # Create split
        node.feature = best_feature
        node.threshold = best_threshold
        
        left_mask = X[:, best_feature] <= best_threshold
        
        # Recursively build children
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """
        Fit the decision tree classifier
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        self._validate_input(X, y)
        
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Build tree
        self.root = self._build_tree(X, y)
        
        self.is_fitted = True
        return self
    
    def _predict_single(self, x: np.ndarray, node: Node) -> int:
        """
        Predict class for a single sample
        
        Args:
            x: Sample features
            node: Current node
            
        Returns:
            Predicted class
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        self._check_fitted()
        self._validate_input(X)
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features, but model was fitted with {self.n_features}")
        
        predictions = []
        for x in X:
            pred = self._predict_single(x, self.root)
            predictions.append(pred)
        
        return np.array(predictions)


class DecisionTreeRegressor(BaseRegressor):
    """
    Decision Tree Regressor using CART algorithm
    """
    
    def __init__(self, criterion: str = 'squared_error', max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: Optional[Union[int, str]] = None,
                 random_state: Optional[int] = None):
        """
        Initialize Decision Tree Regressor
        
        Args:
            criterion: Splitting criterion ('squared_error', 'absolute_error')
            max_depth: Maximum depth of tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            random_state: Random seed
        """
        super().__init__()
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
        self.root = None
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """
        Calculate impurity based on criterion
        
        Args:
            y: Target values
            
        Returns:
            Impurity value
        """
        if len(y) == 0:
            return 0
        
        if self.criterion == 'squared_error':
            return self._mse(y)
        elif self.criterion == 'absolute_error':
            return self._mae(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _mse(self, y: np.ndarray) -> float:
        """Calculate Mean Squared Error"""
        if len(y) == 0:
            return 0
        
        mean_y = np.mean(y)
        mse = np.mean((y - mean_y) ** 2)
        return mse
    
    def _mae(self, y: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        if len(y) == 0:
            return 0
        
        median_y = np.median(y)
        mae = np.mean(np.abs(y - median_y))
        return mae
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Find the best split for the data
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (best_feature, best_threshold, best_reduction)
        """
        n_features = X.shape[1]
        
        # Determine features to consider
        if self.max_features is None:
            features_to_consider = range(n_features)
        elif isinstance(self.max_features, int):
            if self.random_state is not None:
                np.random.seed(self.random_state)
            features_to_consider = np.random.choice(n_features, 
                                                  min(self.max_features, n_features), 
                                                  replace=False)
        elif self.max_features == 'sqrt':
            n_features_to_consider = int(np.sqrt(n_features))
            if self.random_state is not None:
                np.random.seed(self.random_state)
            features_to_consider = np.random.choice(n_features, 
                                                  n_features_to_consider, 
                                                  replace=False)
        else:
            features_to_consider = range(n_features)
        
        best_reduction = -1
        best_feature = None
        best_threshold = None
        
        parent_impurity = self._calculate_impurity(y)
        
        for feature in features_to_consider:
            feature_values = X[:, feature]
            unique_values = np.unique(feature_values)
            
            # Try thresholds between unique values
            thresholds = []
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                thresholds.append(threshold)
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                
                # Check if split creates valid children
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(~left_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate weighted impurity reduction
                y_left = y[left_mask]
                y_right = y[~left_mask]
                
                n = len(y)
                n_left, n_right = len(y_left), len(y_right)
                
                left_impurity = self._calculate_impurity(y_left)
                right_impurity = self._calculate_impurity(y_right)
                
                weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
                reduction = parent_impurity - weighted_impurity
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_reduction
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively build the decision tree
        
        Args:
            X: Feature matrix
            y: Target vector
            depth: Current depth
            
        Returns:
            Root node of the tree
        """
        n_samples = X.shape[0]
        
        # Create node
        node = Node(samples=n_samples, impurity=self._calculate_impurity(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            # Create leaf node
            if self.criterion == 'squared_error':
                node.value = np.mean(y)
            else:  # absolute_error
                node.value = np.median(y)
            return node
        
        # Find best split
        best_feature, best_threshold, best_reduction = self._best_split(X, y)
        
        if best_feature is None or best_reduction <= 0:
            # Create leaf node if no good split found
            if self.criterion == 'squared_error':
                node.value = np.mean(y)
            else:  # absolute_error
                node.value = np.median(y)
            return node
        
        # Create split
        node.feature = best_feature
        node.threshold = best_threshold
        
        left_mask = X[:, best_feature] <= best_threshold
        
        # Recursively build children
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        """
        Fit the decision tree regressor
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        self._validate_input(X, y)
        
        self.n_features = X.shape[1]
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Build tree
        self.root = self._build_tree(X, y)
        
        self.is_fitted = True
        return self
    
    def _predict_single(self, x: np.ndarray, node: Node) -> float:
        """
        Predict value for a single sample
        
        Args:
            x: Sample features
            node: Current node
            
        Returns:
            Predicted value
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        self._check_fitted()
        self._validate_input(X)
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features, but model was fitted with {self.n_features}")
        
        predictions = []
        for x in X:
            pred = self._predict_single(x, self.root)
            predictions.append(pred)
        
        return np.array(predictions)
