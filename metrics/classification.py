"""
Classification metrics for model evaluation
"""
import numpy as np
from typing import List, Optional


class ClassificationMetrics:
    """
    Collection of classification evaluation metrics
    """
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy score
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score (fraction of correct predictions)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary', 
                  pos_label: int = 1) -> float:
        """
        Calculate precision score
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy ('binary', 'macro', 'micro')
            pos_label: Positive class label for binary classification
            
        Returns:
            Precision score
        """
        if average == 'binary':
            return ClassificationMetrics._binary_precision(y_true, y_pred, pos_label)
        elif average == 'macro':
            return ClassificationMetrics._macro_precision(y_true, y_pred)
        elif average == 'micro':
            return ClassificationMetrics._micro_precision(y_true, y_pred)
        else:
            raise ValueError("average must be 'binary', 'macro', or 'micro'")
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary', 
               pos_label: int = 1) -> float:
        """
        Calculate recall score
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy ('binary', 'macro', 'micro')
            pos_label: Positive class label for binary classification
            
        Returns:
            Recall score
        """
        if average == 'binary':
            return ClassificationMetrics._binary_recall(y_true, y_pred, pos_label)
        elif average == 'macro':
            return ClassificationMetrics._macro_recall(y_true, y_pred)
        elif average == 'micro':
            return ClassificationMetrics._micro_recall(y_true, y_pred)
        else:
            raise ValueError("average must be 'binary', 'macro', or 'micro'")
    
    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary', 
                 pos_label: int = 1) -> float:
        """
        Calculate F1 score
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy ('binary', 'macro', 'micro')
            pos_label: Positive class label for binary classification
            
        Returns:
            F1 score
        """
        prec = ClassificationMetrics.precision(y_true, y_pred, average, pos_label)
        rec = ClassificationMetrics.recall(y_true, y_pred, average, pos_label)
        
        if prec + rec == 0:
            return 0.0
        
        return 2 * (prec * rec) / (prec + rec)
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix of shape (n_classes, n_classes)
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        
        return matrix
    
    @staticmethod
    def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Generate a comprehensive classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing precision, recall, f1-score for each class
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        report = {}
        
        for cls in classes:
            # Convert to binary classification for this class
            binary_true = (y_true == cls).astype(int)
            binary_pred = (y_pred == cls).astype(int)
            
            prec = ClassificationMetrics._binary_precision(binary_true, binary_pred, 1)
            rec = ClassificationMetrics._binary_recall(binary_true, binary_pred, 1)
            f1 = ClassificationMetrics.f1_score(binary_true, binary_pred, 'binary', 1)
            support = np.sum(y_true == cls)
            
            report[f'class_{cls}'] = {
                'precision': prec,
                'recall': rec,
                'f1-score': f1,
                'support': support
            }
        
        # Overall metrics
        report['accuracy'] = ClassificationMetrics.accuracy(y_true, y_pred)
        report['macro_avg'] = {
            'precision': ClassificationMetrics.precision(y_true, y_pred, 'macro'),
            'recall': ClassificationMetrics.recall(y_true, y_pred, 'macro'),
            'f1-score': ClassificationMetrics.f1_score(y_true, y_pred, 'macro')
        }
        
        return report
    
    # Helper methods for different averaging strategies
    @staticmethod
    def _binary_precision(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int) -> float:
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        
        if tp + fp == 0:
            return 0.0
        
        return tp / (tp + fp)
    
    @staticmethod
    def _binary_recall(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int) -> float:
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        
        if tp + fn == 0:
            return 0.0
        
        return tp / (tp + fn)
    
    @staticmethod
    def _macro_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        classes = np.unique(y_true)
        precisions = []
        
        for cls in classes:
            binary_true = (y_true == cls).astype(int)
            binary_pred = (y_pred == cls).astype(int)
            precisions.append(ClassificationMetrics._binary_precision(binary_true, binary_pred, 1))
        
        return np.mean(precisions)
    
    @staticmethod
    def _macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        classes = np.unique(y_true)
        recalls = []
        
        for cls in classes:
            binary_true = (y_true == cls).astype(int)
            binary_pred = (y_pred == cls).astype(int)
            recalls.append(ClassificationMetrics._binary_recall(binary_true, binary_pred, 1))
        
        return np.mean(recalls)
    
    @staticmethod
    def _micro_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # For micro-averaging, precision equals accuracy
        return ClassificationMetrics.accuracy(y_true, y_pred)
    
    @staticmethod
    def _micro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # For micro-averaging, recall equals accuracy
        return ClassificationMetrics.accuracy(y_true, y_pred)
