"""
Data Scaling Classes

Scaling transforms features to have similar ranges, which helps optimizers converge faster
and prevents features with larger magnitudes from dominating the learning process.

Mathematical Formulas:

Standard Scaler (Z-score normalization):
X_scaled = (X - μ) / σ
Where μ = mean, σ = standard deviation

Min-Max Scaler:
X_scaled = (X - X_min) / (X_max - X_min)

Use Cases:
- Before training any ML model (especially with gradient descent)
- When features have different units/scales
- Neural networks (almost always needed)
- Distance-based algorithms (KNN, clustering)

When NOT to use:
- Tree-based models (less sensitive to scale)
- When features already have similar scales
"""

import numpy as np


class StandardScaler:
    """Standard Scaler (Z-score Normalization)
    
    Transforms features to have mean=0 and std=1.
    Good when data follows normal distribution.
    """
    
    def __init__(self):
        """Initialize Standard Scaler"""
        self.mean_ = None
        self.std_ = None
        self.fitted = False
        self.name = "StandardScaler"
        
    def fit(self, X):
        """
        Learn the mean and standard deviation from training data
        
        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features)
        """
        X = np.array(X)
        
        # Calculate mean and std for each feature
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        
        # Avoid division by zero for constant features
        self.std_ = np.where(self.std_ == 0, 1, self.std_)
        
        self.fitted = True
        
    def transform(self, X):
        """
        Apply scaling transformation
        
        Args:
            X (np.ndarray): Data to transform
            
        Returns:
            np.ndarray: Scaled data
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        X = np.array(X)
        
        # Apply z-score normalization: (X - μ) / σ
        X_scaled = (X - self.mean_) / self.std_
        
        return X_scaled
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Reverse the scaling transformation
        
        Args:
            X_scaled (np.ndarray): Scaled data
            
        Returns:
            np.ndarray: Original scale data
        """
        if not self.fitted:
            raise ValueError("Must call fit() before inverse_transform()")
        
        X_scaled = np.array(X_scaled)
        
        # Reverse: X = X_scaled * σ + μ
        X_original = X_scaled * self.std_ + self.mean_
        
        return X_original
    
    def __str__(self):
        return f"StandardScaler(fitted={self.fitted})"
    
    def __repr__(self):
        return "StandardScaler()"


class MinMaxScaler:
    """Min-Max Scaler
    
    Transforms features to a fixed range, typically [0, 1].
    Good when you know the approximate upper/lower bounds.
    """
    
    def __init__(self, feature_range=(0, 1)):
        """
        Initialize Min-Max Scaler
        
        Args:
            feature_range (tuple): Target range for scaling
        """
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.fitted = False
        self.name = f"MinMaxScaler({feature_range})"
        
    def fit(self, X):
        """
        Learn the min and max from training data
        
        Args:
            X (np.ndarray): Training data
        """
        X = np.array(X)
        
        # Calculate min and max for each feature
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        
        # Avoid division by zero for constant features
        self.range_ = self.max_ - self.min_
        self.range_ = np.where(self.range_ == 0, 1, self.range_)
        
        self.fitted = True
        
    def transform(self, X):
        """
        Apply min-max scaling
        
        Args:
            X (np.ndarray): Data to transform
            
        Returns:
            np.ndarray: Scaled data
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        X = np.array(X)
        
        # Apply min-max scaling: (X - min) / (max - min)
        X_std = (X - self.min_) / self.range_
        
        # Scale to target range
        scale = self.feature_range[1] - self.feature_range[0]
        X_scaled = X_std * scale + self.feature_range[0]
        
        return X_scaled
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Reverse the min-max scaling
        
        Args:
            X_scaled (np.ndarray): Scaled data
            
        Returns:
            np.ndarray: Original scale data
        """
        if not self.fitted:
            raise ValueError("Must call fit() before inverse_transform()")
        
        X_scaled = np.array(X_scaled)
        
        # Reverse scaling
        scale = self.feature_range[1] - self.feature_range[0]
        X_std = (X_scaled - self.feature_range[0]) / scale
        X_original = X_std * self.range_ + self.min_
        
        return X_original
    
    def __str__(self):
        return f"MinMaxScaler(range={self.feature_range}, fitted={self.fitted})"
    
    def __repr__(self):
        return f"MinMaxScaler(feature_range={self.feature_range})"


# Example usage and educational demonstration
if __name__ == "__main__":
    print("=== Data Scaling Educational Examples ===\n")
    
    # Create sample data with different scales
    np.random.seed(42)
    data = np.array([
        [1000, 0.1, 50],      # Feature 1: large scale
        [1200, 0.3, 75],      # Feature 2: small scale  
        [800, 0.05, 30],      # Feature 3: medium scale
        [1500, 0.8, 90],
        [900, 0.2, 40]
    ])
    
    print("Original Data:")
    print("Features: [Income($), Score(0-1), Age(years)]")
    print(data)
    print(f"Means: {np.mean(data, axis=0)}")
    print(f"Stds:  {np.std(data, axis=0)}")
    print(f"Mins:  {np.min(data, axis=0)}")
    print(f"Maxs:  {np.max(data, axis=0)}")
    
    # Example 1: Standard Scaler
    print("\n" + "="*50)
    print("Example 1: Standard Scaler")
    
    scaler_std = StandardScaler()
    data_std = scaler_std.fit_transform(data)
    
    print("After Standard Scaling:")
    print(data_std)
    print(f"Means: {np.mean(data_std, axis=0)}")
    print(f"Stds:  {np.std(data_std, axis=0)}")
    
    # Test inverse transform
    data_recovered = scaler_std.inverse_transform(data_std)
    print("Recovered original data (should match):")
    print(np.allclose(data, data_recovered))
    
    # Example 2: Min-Max Scaler
    print("\n" + "="*50)
    print("Example 2: Min-Max Scaler")
    
    scaler_mm = MinMaxScaler()
    data_mm = scaler_mm.fit_transform(data)
    
    print("After Min-Max Scaling [0,1]:")
    print(data_mm)
    print(f"Mins: {np.min(data_mm, axis=0)}")
    print(f"Maxs: {np.max(data_mm, axis=0)}")
    
    # Example 3: Custom range
    print("\n" + "="*50)
    print("Example 3: Min-Max Scaler [-1, 1]")
    
    scaler_custom = MinMaxScaler(feature_range=(-1, 1))
    data_custom = scaler_custom.fit_transform(data)
    
    print("After Min-Max Scaling [-1,1]:")
    print(data_custom)
    print(f"Mins: {np.min(data_custom, axis=0)}")
    print(f"Maxs: {np.max(data_custom, axis=0)}")
    
    # Example 4: Impact on gradient descent
    print("\n" + "="*50)
    print("Example 4: Why Scaling Matters for Gradient Descent")
    
    # Simulate gradient computation on unscaled vs scaled data
    print("Unscaled data ranges:", np.max(data, axis=0) - np.min(data, axis=0))
    print("Scaled data ranges:  ", np.max(data_std, axis=0) - np.min(data_std, axis=0))
    
    print("\nKey insights:")
    print("✅ Standard Scaler: Use when data is roughly normal")
    print("✅ Min-Max Scaler: Use when you know bounds")
    print("✅ Both prevent large-scale features from dominating")
    print("✅ Critical for gradient descent convergence")
    print("⚠️  Always fit on training data only!")
    print("⚠️  Apply same transformation to test data")
