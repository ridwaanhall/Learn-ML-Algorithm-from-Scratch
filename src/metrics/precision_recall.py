"""
Precision and Recall metrics implementation.

This module implements precision and recall metrics for classification problems.
Precision measures the proportion of positive predictions that were actually correct.
Recall measures the proportion of actual positives that were correctly identified.
"""
import numpy as np

class Precision:
    """
    Precision metric for binary and multiclass classification.
    
    Precision = True Positives / (True Positives + False Positives)
    """
    
    def __init__(self, average='binary'):
        """
        Initialize precision metric.
        
        Args:
            average (str): Type of averaging ('binary', 'macro', 'micro')
        """
        self.average = average
    
    def calculate(self, y_true, y_pred):
        """
        Calculate precision score.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            float: Precision score
        """
        if self.average == 'binary':
            return self._binary_precision(y_true, y_pred)
        elif self.average == 'macro':
            return self._macro_precision(y_true, y_pred)
        elif self.average == 'micro':
            return self._micro_precision(y_true, y_pred)
        else:
            raise ValueError(f"Unknown average type: {self.average}")
    
    def _binary_precision(self, y_true, y_pred):
        """Calculate precision for binary classification."""
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        
        if predicted_positives == 0:
            return 0.0
        
        return true_positives / predicted_positives
    
    def _macro_precision(self, y_true, y_pred):
        """Calculate macro-averaged precision."""
        classes = np.unique(y_true)
        precisions = []
        
        for cls in classes:
            # Convert to binary problem for each class
            binary_true = (y_true == cls).astype(int)
            binary_pred = (y_pred == cls).astype(int)
            precision = self._binary_precision(binary_true, binary_pred)
            precisions.append(precision)
        
        return np.mean(precisions)
    
    def _micro_precision(self, y_true, y_pred):
        """Calculate micro-averaged precision."""
        return np.sum(y_true == y_pred) / len(y_true)

class Recall:
    """
    Recall metric for binary and multiclass classification.
    
    Recall = True Positives / (True Positives + False Negatives)
    """
    
    def __init__(self, average='binary'):
        """
        Initialize recall metric.
        
        Args:
            average (str): Type of averaging ('binary', 'macro', 'micro')
        """
        self.average = average
    
    def calculate(self, y_true, y_pred):
        """
        Calculate recall score.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            float: Recall score
        """
        if self.average == 'binary':
            return self._binary_recall(y_true, y_pred)
        elif self.average == 'macro':
            return self._macro_recall(y_true, y_pred)
        elif self.average == 'micro':
            return self._micro_recall(y_true, y_pred)
        else:
            raise ValueError(f"Unknown average type: {self.average}")
    
    def _binary_recall(self, y_true, y_pred):
        """Calculate recall for binary classification."""
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        
        if actual_positives == 0:
            return 0.0
        
        return true_positives / actual_positives
    
    def _macro_recall(self, y_true, y_pred):
        """Calculate macro-averaged recall."""
        classes = np.unique(y_true)
        recalls = []
        
        for cls in classes:
            # Convert to binary problem for each class
            binary_true = (y_true == cls).astype(int)
            binary_pred = (y_pred == cls).astype(int)
            recall = self._binary_recall(binary_true, binary_pred)
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def _micro_recall(self, y_true, y_pred):
        """Calculate micro-averaged recall."""
        return np.sum(y_true == y_pred) / len(y_true)
