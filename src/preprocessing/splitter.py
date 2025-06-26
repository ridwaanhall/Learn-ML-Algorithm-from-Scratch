"""
Data splitting utilities for train/validation/test splits.

This module provides functions to split datasets into training,
validation, and testing sets with various strategies.
"""
import numpy as np

class TrainTestSplit:
    """
    Split datasets into train and test sets.
    """
    
    def __init__(self, test_size=0.2, random_state=None, shuffle=True):
        """
        Initialize train-test splitter.
        
        Args:
            test_size (float): Proportion of dataset for test set (0.0 to 1.0)
            random_state (int): Random seed for reproducibility
            shuffle (bool): Whether to shuffle data before splitting
        """
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def split(self, X, y=None):
        """
        Split data into train and test sets.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray, optional): Target labels
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) if y is provided,
                   (X_train, X_test) otherwise
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # Create indices
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Calculate split point
        n_test = int(n_samples * self.test_size)
        n_train = n_samples - n_test
        
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Split features
        X_train = X[train_indices]
        X_test = X[test_indices]
        
        if y is not None:
            y = np.asarray(y)
            y_train = y[train_indices]
            y_test = y[test_indices]
            return X_train, X_test, y_train, y_test
        
        return X_train, X_test

class KFold:
    """
    K-Fold cross-validation splitter.
    """
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        """
        Initialize K-Fold splitter.
        
        Args:
            n_splits (int): Number of folds
            shuffle (bool): Whether to shuffle data before splitting
            random_state (int): Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def split(self, X, y=None):
        """
        Generate K-Fold splits.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray, optional): Target labels (not used but kept for compatibility)
            
        Yields:
            tuple: (train_indices, test_indices) for each fold
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # Create indices
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Calculate fold sizes
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop

class StratifiedSplit:
    """
    Stratified splitting that maintains class distribution.
    """
    
    def __init__(self, test_size=0.2, random_state=None):
        """
        Initialize stratified splitter.
        
        Args:
            test_size (float): Proportion of dataset for test set
            random_state (int): Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def split(self, X, y):
        """
        Split data maintaining class distribution.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target labels
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        classes, class_counts = np.unique(y, return_counts=True)
        
        train_indices = []
        test_indices = []
        
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            np.random.shuffle(cls_indices)
            
            n_test_cls = int(len(cls_indices) * self.test_size)
            
            test_indices.extend(cls_indices[:n_test_cls])
            train_indices.extend(cls_indices[n_test_cls:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        # Shuffle the final indices
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test

def train_test_split(X, y=None, test_size=0.2, random_state=None, shuffle=True):
    """
    Convenience function for train-test splitting.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray, optional): Target labels
        test_size (float): Proportion of dataset for test set
        random_state (int): Random seed for reproducibility
        shuffle (bool): Whether to shuffle data before splitting
        
    Returns:
        tuple: Split data
    """
    splitter = TrainTestSplit(test_size=test_size, 
                             random_state=random_state, 
                             shuffle=shuffle)
    return splitter.split(X, y)
