"""
K-Nearest Neighbors (KNN) implementation from scratch
"""
import numpy as np
from .base_model import BaseClassifier, BaseRegressor
from ..utils.matrix import MatrixUtils
from typing import Union, Optional, Callable
from collections import Counter


class KNNClassifier(BaseClassifier):
    """
    K-Nearest Neighbors classifier
    
    Classifies samples based on the majority class among k nearest neighbors
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform', 
                 metric: str = 'euclidean', p: int = 2):
        """
        Initialize KNN Classifier
        
        Args:
            n_neighbors: Number of neighbors to consider
            weights: Weight function ('uniform', 'distance')
            metric: Distance metric ('euclidean', 'manhattan', 'minkowski')
            p: Parameter for Minkowski distance
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        
        # Training data
        self.X_train = None
        self.y_train = None
        
        # Distance function
        self._distance_func = self._get_distance_function()
    
    def _get_distance_function(self) -> Callable:
        """Get distance function based on metric"""
        if self.metric == 'euclidean':
            return self._euclidean_distance
        elif self.metric == 'manhattan':
            return self._manhattan_distance
        elif self.metric == 'minkowski':
            return self._minkowski_distance
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Manhattan distance between two points"""
        return np.sum(np.abs(x1 - x2))
    
    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Minkowski distance between two points"""
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifier':
        """
        Fit the KNN classifier (store training data)
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        self._validate_input(X, y)
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]
        
        self.is_fitted = True
        return self
    
    def _get_neighbors(self, x: np.ndarray) -> tuple:
        """
        Get k nearest neighbors for a single point
        
        Args:
            x: Query point
            
        Returns:
            Tuple of (neighbor_indices, distances)
        """
        distances = []
        
        # Calculate distances to all training points
        for i, x_train in enumerate(self.X_train):
            distance = self._distance_func(x, x_train)
            distances.append((distance, i))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.n_neighbors]
        
        neighbor_distances = [d[0] for d in k_nearest]
        neighbor_indices = [d[1] for d in k_nearest]
        
        return neighbor_indices, neighbor_distances
    
    def _predict_single(self, x: np.ndarray) -> Union[int, float]:
        """
        Predict class for a single point
        
        Args:
            x: Query point
            
        Returns:
            Predicted class
        """
        neighbor_indices, neighbor_distances = self._get_neighbors(x)
        neighbor_labels = self.y_train[neighbor_indices]
        
        if self.weights == 'uniform':
            # Simple majority vote
            vote_counts = Counter(neighbor_labels)
            predicted_class = vote_counts.most_common(1)[0][0]
        elif self.weights == 'distance':
            # Distance-weighted voting
            class_weights = {}
            
            for i, label in enumerate(neighbor_labels):
                distance = neighbor_distances[i]
                # Avoid division by zero
                weight = 1 / (distance + 1e-8)
                
                if label not in class_weights:
                    class_weights[label] = 0
                class_weights[label] += weight
            
            # Get class with highest weight
            predicted_class = max(class_weights, key=class_weights.get)
        else:
            raise ValueError(f"Unknown weights: {self.weights}")
        
        return predicted_class
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for multiple points
        
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
            pred = self._predict_single(x)
            predictions.append(pred)
        
        return np.array(predictions)
    
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
        
        probabilities = []
        
        for x in X:
            neighbor_indices, neighbor_distances = self._get_neighbors(x)
            neighbor_labels = self.y_train[neighbor_indices]
            
            # Initialize probability array
            class_probs = np.zeros(self.n_classes)
            
            if self.weights == 'uniform':
                # Count votes for each class
                for label in neighbor_labels:
                    class_idx = np.where(self.classes_ == label)[0][0]
                    class_probs[class_idx] += 1
                
                # Normalize to get probabilities
                class_probs = class_probs / self.n_neighbors
            
            elif self.weights == 'distance':
                # Distance-weighted probabilities
                total_weight = 0
                
                for i, label in enumerate(neighbor_labels):
                    distance = neighbor_distances[i]
                    weight = 1 / (distance + 1e-8)
                    
                    class_idx = np.where(self.classes_ == label)[0][0]
                    class_probs[class_idx] += weight
                    total_weight += weight
                
                # Normalize by total weight
                if total_weight > 0:
                    class_probs = class_probs / total_weight
            
            probabilities.append(class_probs)
        
        return np.array(probabilities)


class KNNRegressor(BaseRegressor):
    """
    K-Nearest Neighbors regressor
    
    Predicts target values as the average of k nearest neighbors
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform', 
                 metric: str = 'euclidean', p: int = 2):
        """
        Initialize KNN Regressor
        
        Args:
            n_neighbors: Number of neighbors to consider
            weights: Weight function ('uniform', 'distance')
            metric: Distance metric ('euclidean', 'manhattan', 'minkowski')
            p: Parameter for Minkowski distance
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        
        # Training data
        self.X_train = None
        self.y_train = None
        
        # Distance function
        self._distance_func = self._get_distance_function()
    
    def _get_distance_function(self) -> Callable:
        """Get distance function based on metric"""
        if self.metric == 'euclidean':
            return self._euclidean_distance
        elif self.metric == 'manhattan':
            return self._manhattan_distance
        elif self.metric == 'minkowski':
            return self._minkowski_distance
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Manhattan distance between two points"""
        return np.sum(np.abs(x1 - x2))
    
    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Minkowski distance between two points"""
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNRegressor':
        """
        Fit the KNN regressor (store training data)
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        self._validate_input(X, y)
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.n_features = X.shape[1]
        
        self.is_fitted = True
        return self
    
    def _get_neighbors(self, x: np.ndarray) -> tuple:
        """
        Get k nearest neighbors for a single point
        
        Args:
            x: Query point
            
        Returns:
            Tuple of (neighbor_indices, distances)
        """
        distances = []
        
        # Calculate distances to all training points
        for i, x_train in enumerate(self.X_train):
            distance = self._distance_func(x, x_train)
            distances.append((distance, i))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.n_neighbors]
        
        neighbor_distances = [d[0] for d in k_nearest]
        neighbor_indices = [d[1] for d in k_nearest]
        
        return neighbor_indices, neighbor_distances
    
    def _predict_single(self, x: np.ndarray) -> float:
        """
        Predict target value for a single point
        
        Args:
            x: Query point
            
        Returns:
            Predicted target value
        """
        neighbor_indices, neighbor_distances = self._get_neighbors(x)
        neighbor_targets = self.y_train[neighbor_indices]
        
        if self.weights == 'uniform':
            # Simple average
            prediction = np.mean(neighbor_targets)
        elif self.weights == 'distance':
            # Distance-weighted average
            weights = []
            for distance in neighbor_distances:
                # Avoid division by zero
                weight = 1 / (distance + 1e-8)
                weights.append(weight)
            
            weights = np.array(weights)
            prediction = np.average(neighbor_targets, weights=weights)
        else:
            raise ValueError(f"Unknown weights: {self.weights}")
        
        return prediction
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for multiple points
        
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
            pred = self._predict_single(x)
            predictions.append(pred)
        
        return np.array(predictions)
