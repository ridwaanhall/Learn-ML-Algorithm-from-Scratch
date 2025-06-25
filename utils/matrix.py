"""
Matrix utilities for ML operations using pure Python and NumPy
"""
import numpy as np
from typing import Union, List, Tuple


class MatrixUtils:
    """Utility class for matrix operations used in ML algorithms"""
    
    @staticmethod
    def dot_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute dot product of two matrices/vectors
        
        Args:
            a: First matrix/vector
            b: Second matrix/vector
            
        Returns:
            Dot product result
        """
        return np.dot(a, b)
    
    @staticmethod
    def transpose(matrix: np.ndarray) -> np.ndarray:
        """
        Transpose a matrix
        
        Args:
            matrix: Input matrix
            
        Returns:
            Transposed matrix
        """
        return matrix.T
    
    @staticmethod
    def add_bias_column(X: np.ndarray) -> np.ndarray:
        """
        Add bias column (column of ones) to feature matrix
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Feature matrix with bias column of shape (n_samples, n_features + 1)
        """
        bias_column = np.ones((X.shape[0], 1))
        return np.concatenate([bias_column, X], axis=1)
    
    @staticmethod
    def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points
        
        Args:
            point1: First point
            point2: Second point
            
        Returns:
            Euclidean distance
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function
        
        Args:
            z: Input values
            
        Returns:
            Sigmoid output
        """
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """
        Softmax activation function
        
        Args:
            z: Input values
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    @staticmethod
    def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Convert labels to one-hot encoding
        
        Args:
            y: Labels array
            num_classes: Number of classes
            
        Returns:
            One-hot encoded labels
        """
        encoded = np.zeros((len(y), num_classes))
        encoded[np.arange(len(y)), y] = 1
        return encoded


class DataUtils:
    """Utility class for data operations"""
    
    @staticmethod
    def shuffle_data(X: np.ndarray, y: np.ndarray, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shuffle data while maintaining X-y correspondence
        
        Args:
            X: Feature matrix
            y: Target vector
            random_state: Random seed for reproducibility
            
        Returns:
            Shuffled X and y
        """
        if random_state:
            np.random.seed(random_state)
        
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]
    
    @staticmethod
    def create_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create mini-batches from data
        
        Args:
            X: Feature matrix
            y: Target vector
            batch_size: Size of each batch
            
        Returns:
            List of (X_batch, y_batch) tuples
        """
        batches = []
        n_samples = len(X)
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]
            y_batch = y[i:end_idx]
            batches.append((X_batch, y_batch))
            
        return batches
