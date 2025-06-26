"""
F1 Score metric implementation.

This module implements the F1 score metric, which is the harmonic mean
of precision and recall. It provides a balanced measure of a model's
performance by considering both false positives and false negatives.
"""
import numpy as np
from .precision_recall import Precision, Recall

class F1Score:
    """
    F1 Score metric for binary and multiclass classification.
    
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    """
    
    def __init__(self, average='binary'):
        """
        Initialize F1 score metric.
        
        Args:
            average (str): Type of averaging ('binary', 'macro', 'micro')
        """
        self.average = average
        self.precision = Precision(average=average)
        self.recall = Recall(average=average)
    
    def calculate(self, y_true, y_pred):
        """
        Calculate F1 score.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            float: F1 score
        """
        precision = self.precision.calculate(y_true, y_pred)
        recall = self.recall.calculate(y_true, y_pred)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_with_components(self, y_true, y_pred):
        """
        Calculate F1 score along with precision and recall components.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            dict: Dictionary containing f1, precision, and recall scores
        """
        precision = self.precision.calculate(y_true, y_pred)
        recall = self.recall.calculate(y_true, y_pred)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }
