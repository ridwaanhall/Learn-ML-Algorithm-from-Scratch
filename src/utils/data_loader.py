"""
Data Loading Utilities

This module provides convenient functions for loading and generating datasets
for machine learning experiments and education.
"""

import numpy as np


class DataLoader:
    """Data Loading and Generation Utilities
    
    Provides methods to generate synthetic datasets for educational purposes
    and load common dataset formats.
    """
    
    @staticmethod
    def generate_linear_regression_data(n_samples=100, n_features=1, noise_level=0.1, 
                                      coefficients=None, bias=0.0, random_seed=None):
        """
        Generate synthetic linear regression data
        
        Args:
            n_samples (int): Number of samples to generate
            n_features (int): Number of features
            noise_level (float): Standard deviation of noise
            coefficients (list): True coefficients (if None, random)
            bias (float): True bias term
            random_seed (int): Random seed for reproducibility
            
        Returns:
            tuple: (X, y, true_coefficients, true_bias)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate random features
        X = np.random.uniform(-2, 2, (n_samples, n_features))
        
        # Generate or use provided coefficients
        if coefficients is None:
            true_coefficients = np.random.uniform(-3, 3, n_features)
        else:
            true_coefficients = np.array(coefficients)
            if len(true_coefficients) != n_features:
                raise ValueError("Coefficients length must match n_features")
        
        # Generate target values: y = X @ coefficients + bias + noise
        y = np.dot(X, true_coefficients) + bias
        if noise_level > 0:
            y += np.random.normal(0, noise_level, n_samples)
        
        return X, y, true_coefficients, bias
    
    @staticmethod
    def generate_classification_data(n_samples=100, n_features=2, n_classes=2, 
                                   noise_level=0.1, random_seed=None):
        """
        Generate synthetic classification data
        
        Args:
            n_samples (int): Number of samples
            n_features (int): Number of features
            n_classes (int): Number of classes
            noise_level (float): Amount of noise/overlap
            random_seed (int): Random seed
            
        Returns:
            tuple: (X, y)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate cluster centers
        centers = np.random.uniform(-3, 3, (n_classes, n_features))
        
        # Generate samples around centers
        samples_per_class = n_samples // n_classes
        X = []
        y = []
        
        for class_idx in range(n_classes):
            # Generate samples around this center
            center = centers[class_idx]
            class_samples = np.random.normal(center, noise_level, 
                                           (samples_per_class, n_features))
            X.append(class_samples)
            y.extend([class_idx] * samples_per_class)
        
        # Handle remaining samples
        remaining = n_samples - len(y)
        if remaining > 0:
            center = centers[0]
            extra_samples = np.random.normal(center, noise_level, 
                                           (remaining, n_features))
            X.append(extra_samples)
            y.extend([0] * remaining)
        
        X = np.vstack(X)
        y = np.array(y)
        
        # Shuffle the data
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    @staticmethod
    def train_test_split(X, y, test_size=0.2, random_seed=None):
        """
        Split data into training and testing sets
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Targets
            test_size (float): Fraction for test set
            random_seed (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        # Random shuffle
        indices = np.random.permutation(n_samples)
        
        # Split indices
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        # Split data
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def add_polynomial_features(X, degree=2):
        """
        Add polynomial features to the data
        
        Args:
            X (np.ndarray): Original features
            degree (int): Polynomial degree
            
        Returns:
            np.ndarray: Features with polynomial terms
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Start with original features
        features = [X]
        
        # Add polynomial terms
        for d in range(2, degree + 1):
            poly_features = X ** d
            features.append(poly_features)
        
        # Combine all features
        X_poly = np.hstack(features)
        
        return X_poly


# Example usage and educational demonstration
if __name__ == "__main__":
    print("=== Data Loading Utilities Educational Examples ===\n")
    
    # Example 1: Linear regression data
    print("Example 1: Generate Linear Regression Data")
    
    X, y, true_coef, true_bias = DataLoader.generate_linear_regression_data(
        n_samples=50, 
        n_features=2, 
        coefficients=[2.5, -1.5], 
        bias=3.0,
        noise_level=0.2,
        random_seed=42
    )
    
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    print(f"True equation: y = {true_coef[0]:.1f}*x1 + {true_coef[1]:.1f}*x2 + {true_bias:.1f}")
    print(f"Feature ranges: X1=[{X[:,0].min():.2f}, {X[:,0].max():.2f}], X2=[{X[:,1].min():.2f}, {X[:,1].max():.2f}]")
    print(f"Target range: y=[{y.min():.2f}, {y.max():.2f}]")
    
    # Example 2: Classification data
    print("\n" + "="*50)
    print("Example 2: Generate Classification Data")
    
    X_cls, y_cls = DataLoader.generate_classification_data(
        n_samples=100,
        n_features=2,
        n_classes=3,
        noise_level=0.5,
        random_seed=42
    )
    
    print(f"Generated {len(X_cls)} samples for {len(np.unique(y_cls))} classes")
    print(f"Class distribution: {np.bincount(y_cls)}")
    print(f"Feature ranges: X1=[{X_cls[:,0].min():.2f}, {X_cls[:,0].max():.2f}], X2=[{X_cls[:,1].min():.2f}, {X_cls[:,1].max():.2f}]")
    
    # Example 3: Train-test split
    print("\n" + "="*50)
    print("Example 3: Train-Test Split")
    
    X_train, X_test, y_train, y_test = DataLoader.train_test_split(
        X, y, test_size=0.3, random_seed=42
    )
    
    print(f"Original data: {len(X)} samples")
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Example 4: Polynomial features
    print("\n" + "="*50)
    print("Example 4: Add Polynomial Features")
    
    # Simple 1D data
    X_simple = np.linspace(-2, 2, 10).reshape(-1, 1)
    X_poly = DataLoader.add_polynomial_features(X_simple, degree=3)
    
    print("Original features (first 5 samples):")
    print(X_simple[:5].flatten())
    print(f"With polynomial features up to degree 3 (shape: {X_poly.shape}):")
    print("Features: [x, x², x³]")
    print(X_poly[:5])
    
    print("\nKey utilities provided:")
    print("✅ Generate synthetic regression data with known ground truth")
    print("✅ Generate classification data with controllable complexity")
    print("✅ Proper train-test splitting with random shuffling")
    print("✅ Add polynomial features for non-linear modeling")
    print("💡 Use these for experimentation and understanding algorithms")
