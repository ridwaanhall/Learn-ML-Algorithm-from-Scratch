"""
Data splitting utilities for train/validation/test splits
"""
import numpy as np
from typing import Tuple, Optional, Union


class TrainTestSplit:
    """
    Split arrays or matrices into random train and test subsets
    """
    
    @staticmethod
    def train_test_split(X: np.ndarray, y: Optional[np.ndarray] = None, 
                        test_size: float = 0.25, train_size: Optional[float] = None,
                        random_state: Optional[int] = None, 
                        shuffle: bool = True, stratify: Optional[np.ndarray] = None) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split arrays into random train and test subsets
        
        Args:
            X: Input data to split
            y: Target data to split (optional)
            test_size: Proportion of dataset to include in test split
            train_size: Proportion of dataset to include in train split
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle data before splitting
            stratify: If not None, data is split in a stratified fashion
            
        Returns:
            List containing train-test split of inputs
        """
        if test_size is None and train_size is None:
            test_size = 0.25
        
        n_samples = len(X)
        
        if test_size is not None:
            if not 0 < test_size < 1:
                raise ValueError("test_size must be between 0 and 1")
            n_test = int(n_samples * test_size)
        elif train_size is not None:
            if not 0 < train_size < 1:
                raise ValueError("train_size must be between 0 and 1")
            n_test = n_samples - int(n_samples * train_size)
        
        n_train = n_samples - n_test
        
        if random_state is not None:
            np.random.seed(random_state)
        
        if stratify is not None:
            # Stratified split
            unique_classes, class_counts = np.unique(stratify, return_counts=True)
            train_indices = []
            test_indices = []
            
            for cls in unique_classes:
                cls_indices = np.where(stratify == cls)[0]
                if shuffle:
                    np.random.shuffle(cls_indices)
                
                cls_n_test = int(len(cls_indices) * (n_test / n_samples))
                cls_test_indices = cls_indices[:cls_n_test]
                cls_train_indices = cls_indices[cls_n_test:]
                
                train_indices.extend(cls_train_indices)
                test_indices.extend(cls_test_indices)
            
            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)
            
            if shuffle:
                np.random.shuffle(train_indices)
                np.random.shuffle(test_indices)
        else:
            # Regular split
            indices = np.arange(n_samples)
            if shuffle:
                np.random.shuffle(indices)
            
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        
        if y is not None:
            y_train = y[train_indices]
            y_test = y[test_indices]
            return X_train, X_test, y_train, y_test
        else:
            return X_train, X_test


class KFold:
    """
    K-Folds cross-validator
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        """
        Initialize KFold cross-validator
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate indices to split data into training and test sets
        
        Args:
            X: Input data
            y: Target data (ignored, present for API consistency)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> int:
        """
        Returns the number of splitting iterations
        
        Args:
            X: Input data (ignored)
            y: Target data (ignored)
            
        Returns:
            Number of splits
        """
        return self.n_splits


class StratifiedKFold:
    """
    Stratified K-Folds cross-validator
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        """
        Initialize StratifiedKFold cross-validator
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle each class's samples before splitting
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate indices to split data into training and test sets
        
        Args:
            X: Input data
            y: Target data
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if y is None:
            raise ValueError("y must be provided for stratified split")
        
        unique_classes, class_indices = np.unique(y, return_inverse=True)
        n_samples = len(X)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Create folds for each class
        class_folds = {}
        for cls_idx, cls in enumerate(unique_classes):
            cls_samples = np.where(class_indices == cls_idx)[0]
            
            if self.shuffle:
                np.random.shuffle(cls_samples)
            
            # Distribute class samples across folds
            cls_fold_sizes = np.full(self.n_splits, len(cls_samples) // self.n_splits, dtype=int)
            cls_fold_sizes[:len(cls_samples) % self.n_splits] += 1
            
            cls_folds = []
            current = 0
            for fold_size in cls_fold_sizes:
                cls_folds.append(cls_samples[current:current + fold_size])
                current += fold_size
            
            class_folds[cls_idx] = cls_folds
        
        # Combine folds across classes
        for fold_idx in range(self.n_splits):
            test_indices = []
            train_indices = []
            
            for cls_idx in range(len(unique_classes)):
                cls_folds = class_folds[cls_idx]
                test_indices.extend(cls_folds[fold_idx])
                
                # Train indices are all other folds for this class
                for other_fold_idx in range(self.n_splits):
                    if other_fold_idx != fold_idx:
                        train_indices.extend(cls_folds[other_fold_idx])
            
            yield np.array(train_indices), np.array(test_indices)
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> int:
        """
        Returns the number of splitting iterations
        
        Args:
            X: Input data (ignored)
            y: Target data (ignored)
            
        Returns:
            Number of splits
        """
        return self.n_splits


class TimeSeriesSplit:
    """
    Time Series cross-validator
    """
    
    def __init__(self, n_splits: int = 5, max_train_size: Optional[int] = None, 
                 test_size: Optional[int] = None, gap: int = 0):
        """
        Initialize TimeSeriesSplit cross-validator
        
        Args:
            n_splits: Number of splits
            max_train_size: Maximum size for training set
            test_size: Size of test set
            gap: Number of samples to exclude between train and test
        """
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate indices to split data into training and test sets
        
        Args:
            X: Input data
            y: Target data (ignored)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        test_size = self.test_size if self.test_size is not None else n_samples // (self.n_splits + 1)
        
        test_starts = range(test_size + self.gap, n_samples, test_size)
        test_starts = test_starts[:self.n_splits]
        
        for test_start in test_starts:
            test_end = min(test_start + test_size, n_samples)
            train_end = test_start - self.gap
            
            if self.max_train_size and train_end > self.max_train_size:
                train_start = train_end - self.max_train_size
            else:
                train_start = 0
            
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> int:
        """
        Returns the number of splitting iterations
        
        Args:
            X: Input data (ignored)
            y: Target data (ignored)
            
        Returns:
            Number of splits
        """
        return self.n_splits
