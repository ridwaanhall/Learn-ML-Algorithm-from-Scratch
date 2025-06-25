"""
K-Means Clustering implementation from scratch
"""
import numpy as np
from .base_model import BaseUnsupervised
from ..utils.matrix import MatrixUtils
from typing import Optional, Union


class KMeans(BaseUnsupervised):
    """
    K-Means clustering algorithm
    
    Partitions data into k clusters by minimizing within-cluster sum of squares
    """
    
    def __init__(self, n_clusters: int = 8, init: str = 'k-means++', 
                 max_iter: int = 300, tolerance: float = 1e-4, 
                 random_state: Optional[int] = None, n_init: int = 10):
        """
        Initialize K-Means clustering
        
        Args:
            n_clusters: Number of clusters
            init: Initialization method ('k-means++', 'random')
            max_iter: Maximum number of iterations
            tolerance: Tolerance for convergence
            random_state: Random seed
            n_init: Number of random initializations
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.n_init = n_init
        
        # Model attributes
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = None
    
    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize cluster centroids
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            Initial centroids of shape (n_clusters, n_features)
        """
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            # Random initialization
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            centroids = np.random.uniform(min_vals, max_vals, (self.n_clusters, n_features))
            
        elif self.init == 'k-means++':
            # K-means++ initialization
            centroids = np.zeros((self.n_clusters, n_features))
            
            # Choose first centroid randomly
            centroids[0] = X[np.random.randint(n_samples)]
            
            # Choose remaining centroids
            for i in range(1, self.n_clusters):
                # Calculate distances to nearest centroid for each point
                distances = np.full(n_samples, float('inf'))
                
                for j in range(n_samples):
                    for k in range(i):
                        dist = MatrixUtils.euclidean_distance(X[j], centroids[k])
                        distances[j] = min(distances[j], dist)
                
                # Choose next centroid with probability proportional to squared distance
                distances_squared = distances ** 2
                probabilities = distances_squared / np.sum(distances_squared)
                
                # Handle edge case where all probabilities are 0
                if np.sum(probabilities) == 0:
                    probabilities = np.ones(n_samples) / n_samples
                
                chosen_idx = np.random.choice(n_samples, p=probabilities)
                centroids[i] = X[chosen_idx]
        
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")
        
        return centroids
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each point to the nearest centroid
        
        Args:
            X: Data matrix
            centroids: Current centroids
            
        Returns:
            Cluster assignments
        """
        n_samples = X.shape[0]
        assignments = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            distances = []
            for j in range(self.n_clusters):
                dist = MatrixUtils.euclidean_distance(X[i], centroids[j])
                distances.append(dist)
            
            assignments[i] = np.argmin(distances)
        
        return assignments
    
    def _update_centroids(self, X: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """
        Update centroids based on cluster assignments
        
        Args:
            X: Data matrix
            assignments: Cluster assignments
            
        Returns:
            Updated centroids
        """
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        
        for i in range(self.n_clusters):
            cluster_points = X[assignments == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # If cluster is empty, keep old centroid
                new_centroids[i] = self.cluster_centers_[i] if self.cluster_centers_ is not None else np.zeros(n_features)
        
        return new_centroids
    
    def _calculate_inertia(self, X: np.ndarray, centroids: np.ndarray, assignments: np.ndarray) -> float:
        """
        Calculate within-cluster sum of squares (inertia)
        
        Args:
            X: Data matrix
            centroids: Centroids
            assignments: Cluster assignments
            
        Returns:
            Inertia value
        """
        inertia = 0
        for i in range(len(X)):
            cluster = assignments[i]
            distance = MatrixUtils.euclidean_distance(X[i], centroids[cluster])
            inertia += distance ** 2
        
        return inertia
    
    def _fit_single_run(self, X: np.ndarray) -> tuple:
        """
        Single run of K-means algorithm
        
        Args:
            X: Data matrix
            
        Returns:
            Tuple of (centroids, labels, inertia, n_iter)
        """
        # Initialize centroids
        centroids = self._init_centroids(X)
        
        for iteration in range(self.max_iter):
            # Assign points to clusters
            old_assignments = getattr(self, '_last_assignments', None)
            assignments = self._assign_clusters(X, centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X, assignments)
            
            # Check for convergence
            if old_assignments is not None:
                if np.array_equal(assignments, old_assignments):
                    break
            
            # Check centroid movement
            centroid_shift = 0
            for i in range(self.n_clusters):
                centroid_shift += MatrixUtils.euclidean_distance(centroids[i], new_centroids[i])
            
            if centroid_shift < self.tolerance:
                break
            
            centroids = new_centroids
            self._last_assignments = assignments
        
        # Calculate final inertia
        inertia = self._calculate_inertia(X, centroids, assignments)
        
        return centroids, assignments, inertia, iteration + 1
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'KMeans':
        """
        Fit K-means clustering
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Ignored (for API compatibility)
            
        Returns:
            Self for method chaining
        """
        self._validate_input(X)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot be larger than number of samples")
        
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None
        best_n_iter = 0
        
        # Run K-means multiple times and keep best result
        for run in range(self.n_init):
            centroids, labels, inertia, n_iter = self._fit_single_run(X)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter
        
        # Store best results
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data
        
        Args:
            X: Data to predict of shape (n_samples, n_features)
            
        Returns:
            Cluster labels of shape (n_samples,)
        """
        self._check_fitted()
        self._validate_input(X)
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features, but model was fitted with {self.n_features}")
        
        return self._assign_clusters(X, self.cluster_centers_)
    
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit K-means and return cluster labels
        
        Args:
            X: Data to fit and predict
            y: Ignored (for API compatibility)
            
        Returns:
            Cluster labels
        """
        self.fit(X, y)
        return self.labels_
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X to cluster-distance space
        
        Args:
            X: Data to transform
            
        Returns:
            Distances to each cluster center
        """
        self._check_fitted()
        self._validate_input(X)
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features, but model was fitted with {self.n_features}")
        
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for i in range(n_samples):
            for j in range(self.n_clusters):
                distances[i, j] = MatrixUtils.euclidean_distance(X[i], self.cluster_centers_[j])
        
        return distances
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit K-means and transform X to cluster-distance space
        
        Args:
            X: Data to fit and transform
            y: Ignored (for API compatibility)
            
        Returns:
            Distances to each cluster center
        """
        self.fit(X, y)
        return self.transform(X)
