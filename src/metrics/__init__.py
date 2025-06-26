"""
Metrics Module

This module contains implementations of various evaluation metrics for machine learning models.
Each metric includes:
- Mathematical formula explanation
- Implementation for different problem types
- When to use each metric
- Interpretation guidelines

Metrics help evaluate how well your model is performing.
"""

from .accuracy import Accuracy
from .precision_recall import Precision, Recall
from .f1_score import F1Score
from .r2_score import R2Score

__all__ = ['Accuracy', 'Precision', 'Recall', 'F1Score', 'R2Score']
