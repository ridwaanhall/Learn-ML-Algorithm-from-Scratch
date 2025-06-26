"""
Preprocessing Module

This module contains implementations of various data preprocessing techniques.
Each preprocessing class includes:
- Mathematical explanation
- When to use each technique
- Implementation details
- Best practices

Proper preprocessing is crucial for machine learning model performance.
"""

from .scaler import StandardScaler, MinMaxScaler
from .encoder import LabelEncoder, OneHotEncoder
from .splitter import TrainTestSplit, KFold, StratifiedSplit, train_test_split

__all__ = ['StandardScaler', 'MinMaxScaler', 'LabelEncoder', 'OneHotEncoder', 
           'TrainTestSplit', 'KFold', 'StratifiedSplit', 'train_test_split']
