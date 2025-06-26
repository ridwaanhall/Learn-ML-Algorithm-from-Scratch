"""
Cross Entropy Loss Function

Mathematical Formula:
For Binary Classification:
BCE = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]

For Multiclass Classification:
CE = -(1/n) * Σ Σ y_ij * log(p_ij)

Where:
- n = number of samples
- y = true labels (one-hot encoded for multiclass)
- p = predicted probabilities
- i = sample index, j = class index

Use Cases:
- Binary classification (sigmoid output)
- Multiclass classification (softmax output)
- When you need probability outputs
- Standard choice for classification tasks

When NOT to use:
- Regression problems (use MSE/MAE instead)
- When predictions are not probabilities
- Imbalanced datasets (consider weighted cross entropy)
"""

import numpy as np


class CrossEntropy:
    """Cross Entropy Loss Function
    
    This is the standard loss function for classification tasks.
    It works with probability distributions and heavily penalizes confident wrong predictions.
    """
    
    def __init__(self, binary=True, epsilon=1e-15):
        """
        Initialize Cross Entropy loss function
        
        Args:
            binary (bool): If True, use binary cross entropy. If False, use categorical.
            epsilon (float): Small value to prevent log(0) which causes numerical instability
        """
        self.binary = binary
        self.epsilon = epsilon
        self.name = "Binary Cross Entropy" if binary else "Categorical Cross Entropy"
        
    def _clip_predictions(self, y_pred):
        """
        Clip predictions to prevent log(0) and log(1) which cause numerical issues
        
        Args:
            y_pred (np.ndarray): Predicted probabilities
            
        Returns:
            np.ndarray: Clipped predictions
        """
        return np.clip(y_pred, self.epsilon, 1 - self.epsilon)
    
    def forward(self, y_true, y_pred):
        """
        Calculate the Cross Entropy loss
        
        Args:
            y_true (np.ndarray): True labels
                - Binary: shape (n_samples,) with values in {0, 1}
                - Multiclass: shape (n_samples, n_classes) one-hot encoded
            y_pred (np.ndarray): Predicted probabilities, same shape as y_true
            
        Returns:
            float: Cross entropy loss value
        """
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Clip predictions to prevent numerical instability
        y_pred = self._clip_predictions(y_pred)
        
        if self.binary:
            return self._binary_cross_entropy(y_true, y_pred)
        else:
            return self._categorical_cross_entropy(y_true, y_pred)
    
    def _binary_cross_entropy(self, y_true, y_pred):
        """
        Calculate Binary Cross Entropy
        
        Mathematical Steps:
        1. For each sample: loss_i = -[y_i * log(p_i) + (1-y_i) * log(1-p_i)]
        2. Average over all samples: BCE = (1/n) * Σ loss_i
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        # Binary Cross Entropy formula
        bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(bce)
    
    def _categorical_cross_entropy(self, y_true, y_pred):
        """
        Calculate Categorical Cross Entropy
        
        Mathematical Steps:
        1. For each sample and class: loss_ij = -y_ij * log(p_ij)
        2. Sum over classes for each sample: loss_i = Σ_j loss_ij
        3. Average over samples: CE = (1/n) * Σ_i loss_i
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        # Categorical Cross Entropy formula
        ce = -np.sum(y_true * np.log(y_pred), axis=1)
        return np.mean(ce)
    
    def backward(self, y_true, y_pred):
        """
        Calculate the gradient of Cross Entropy with respect to predictions
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted probabilities
            
        Returns:
            np.ndarray: Gradient of Cross Entropy with respect to y_pred
        """
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Clip predictions to prevent numerical instability
        y_pred = self._clip_predictions(y_pred)
        
        if self.binary:
            return self._binary_cross_entropy_gradient(y_true, y_pred)
        else:
            return self._categorical_cross_entropy_gradient(y_true, y_pred)
    
    def _binary_cross_entropy_gradient(self, y_true, y_pred):
        """
        Calculate Binary Cross Entropy gradient
        
        Mathematical Derivation:
        BCE = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]
        
        ∂BCE/∂p = -(1/n) * Σ[y/p - (1-y)/(1-p)]
                = -(1/n) * Σ[(y - p) / (p*(1-p))]
                = (1/n) * Σ[(p - y) / (p*(1-p))]
        """
        n = y_true.shape[0]
        gradient = (y_pred - y_true) / (y_pred * (1 - y_pred))
        return gradient / n
    
    def _categorical_cross_entropy_gradient(self, y_true, y_pred):
        """
        Calculate Categorical Cross Entropy gradient
        
        Mathematical Derivation:
        CE = -(1/n) * Σ Σ y_ij * log(p_ij)
        
        ∂CE/∂p_ij = -(1/n) * y_ij / p_ij
                   = (1/n) * (p_ij - y_ij) / p_ij  [when combined with softmax]
        
        For softmax output, the gradient simplifies to:
        ∂CE/∂p_ij = (1/n) * (p_ij - y_ij)
        """
        n = y_true.shape[0]
        # Simplified gradient for softmax + cross entropy
        gradient = (y_pred - y_true) / n
        return gradient
    
    def __call__(self, y_true, y_pred):
        """Allow the class to be called as a function"""
        return self.forward(y_true, y_pred)
    
    def __str__(self):
        if self.binary:
            return "Binary Cross Entropy: L = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]"
        else:
            return "Categorical Cross Entropy: L = -(1/n) * Σ Σ y_ij * log(p_ij)"
    
    def __repr__(self):
        return f"CrossEntropy(binary={self.binary}, epsilon={self.epsilon})"


# Example usage and educational demonstration
if __name__ == "__main__":
    # Example 1: Binary Classification
    print("=== Binary Cross Entropy Examples ===")
    bce_loss = CrossEntropy(binary=True)
    
    # Perfect predictions
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.999, 0.001, 0.999, 0.001])  # Very confident and correct
    
    loss = bce_loss(y_true, y_pred)
    gradient = bce_loss.backward(y_true, y_pred)
    
    print("Example 1a: Very Confident Correct Predictions")
    print(f"True labels: {y_true}")
    print(f"Predictions: {y_pred}")
    print(f"BCE Loss: {loss:.6f}")
    print(f"Gradient: {gradient}\n")
    
    # Confident wrong predictions
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.001, 0.999, 0.001, 0.999])  # Very confident but wrong
    
    loss = bce_loss(y_true, y_pred)
    gradient = bce_loss.backward(y_true, y_pred)
    
    print("Example 1b: Very Confident Wrong Predictions")
    print(f"True labels: {y_true}")
    print(f"Predictions: {y_pred}")
    print(f"BCE Loss: {loss:.6f}")
    print(f"Gradient: {gradient}")
    print("Note: Loss is much higher for confident wrong predictions!\n")
    
    # Example 2: Multiclass Classification
    print("=== Categorical Cross Entropy Examples ===")
    cce_loss = CrossEntropy(binary=False)
    
    # One-hot encoded true labels (3 classes)
    y_true = np.array([
        [1, 0, 0],  # Class 0
        [0, 1, 0],  # Class 1
        [0, 0, 1],  # Class 2
    ])
    
    # Good predictions
    y_pred = np.array([
        [0.8, 0.1, 0.1],  # Correct class has high probability
        [0.1, 0.8, 0.1],  # Correct class has high probability
        [0.1, 0.1, 0.8],  # Correct class has high probability
    ])
    
    loss = cce_loss(y_true, y_pred)
    gradient = cce_loss.backward(y_true, y_pred)
    
    print("Example 2a: Good Multiclass Predictions")
    print(f"True labels:\n{y_true}")
    print(f"Predictions:\n{y_pred}")
    print(f"CCE Loss: {loss:.6f}")
    print(f"Gradient:\n{gradient}\n")
    
    # Bad predictions
    y_pred_bad = np.array([
        [0.1, 0.8, 0.1],  # Wrong class has high probability
        [0.8, 0.1, 0.1],  # Wrong class has high probability
        [0.1, 0.8, 0.1],  # Wrong class has high probability
    ])
    
    loss_bad = cce_loss(y_true, y_pred_bad)
    
    print("Example 2b: Bad Multiclass Predictions")
    print(f"True labels:\n{y_true}")
    print(f"Predictions:\n{y_pred_bad}")
    print(f"CCE Loss: {loss_bad:.6f}")
    print(f"Note: Loss increased from {loss:.6f} to {loss_bad:.6f}")
