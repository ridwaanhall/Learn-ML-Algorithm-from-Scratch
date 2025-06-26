"""
Learn ML Algorithms from Scratch

A comprehensive educational package for understanding machine learning algorithms
by implementing them from scratch using only NumPy.

Author: Ridwan Hall (ridwaanhall.com)
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Ridwan Hall"
__email__ = "contact@ridwaanhall.com"
__url__ = "https://ridwaanhall.com"
__license__ = "MIT"

# Import main modules for easy access
from . import models
from . import optimization
from . import loss_functions
from . import activation_functions
from . import metrics
from . import preprocessing
from . import utils

__all__ = [
    'models',
    'optimization', 
    'loss_functions',
    'activation_functions',
    'metrics',
    'preprocessing',
    'utils'
]
