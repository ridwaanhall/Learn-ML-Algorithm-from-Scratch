"""
Accuracy Metric

Mathematical Formula:
Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)
         = (TP + TN) / (TP + TN + FP + FN)

Where:
- TP = True Positives
- TN = True Negatives  
- FP = False Positives
- FN = False Negatives

Use Cases:
- Classification problems with balanced classes
- When all classes are equally important
- Binary and multiclass classification
- Quick overall performance assessment

When NOT to use:
- Imbalanced datasets (use precision/recall/F1)
- When false positives and false negatives have different costs
- Regression problems (use R², MSE, MAE)

Range: [0, 1] where 1 is perfect accuracy
"""

import numpy as np


class Accuracy:
    """Accuracy Metric for Classification
    
    Measures the fraction of predictions that match the true labels.
    Simple and intuitive but can be misleading with imbalanced data.
    """
    
    def __init__(self):
        """Initialize Accuracy metric"""
        self.name = "Accuracy"
        
    def calculate(self, y_true, y_pred):
        """
        Calculate accuracy score
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels (not probabilities)
            
        Returns:
            float: Accuracy score between 0 and 1
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        # Count correct predictions
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        
        accuracy = correct / total
        return accuracy
    
    def __call__(self, y_true, y_pred):
        """Allow the class to be called as a function"""
        return self.calculate(y_true, y_pred)
    
    def __str__(self):
        return "Accuracy = Correct Predictions / Total Predictions"
    
    def __repr__(self):
        return "Accuracy()"


# Example usage and educational demonstration
if __name__ == "__main__":
    accuracy = Accuracy()
    
    print("=== Accuracy Metric Educational Examples ===\n")
    
    # Example 1: Perfect accuracy
    print("Example 1: Perfect Accuracy")
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0])  # All correct
    
    acc = accuracy(y_true, y_pred)
    print(f"True:     {y_true}")
    print(f"Pred:     {y_pred}")
    print(f"Accuracy: {acc:.3f} (100% correct)")
    
    # Example 2: Partial accuracy
    print("\nExample 2: Partial Accuracy")
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1])  # 3 out of 5 correct
    
    acc = accuracy(y_true, y_pred)
    correct_mask = y_true == y_pred
    print(f"True:     {y_true}")
    print(f"Pred:     {y_pred}")
    print(f"Correct:  {correct_mask}")
    print(f"Accuracy: {acc:.3f} (3/5 = 60%)")
    
    # Example 3: Multiclass accuracy
    print("\nExample 3: Multiclass Classification")
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 2, 2, 0, 1])  # 4 out of 6 correct
    
    acc = accuracy(y_true, y_pred)
    print(f"True:     {y_true}")
    print(f"Pred:     {y_pred}")
    print(f"Accuracy: {acc:.3f} (4/6 = 67%)")
    
    # Example 4: Imbalanced data problem
    print("\nExample 4: Imbalanced Data Problem")
    # 90% class 0, 10% class 1
    y_true = np.array([0]*90 + [1]*10)
    y_pred_naive = np.array([0]*100)  # Naive classifier: always predict 0
    y_pred_good = np.array([0]*85 + [1]*5 + [0]*5 + [1]*5)  # Better classifier
    
    acc_naive = accuracy(y_true, y_pred_naive)
    acc_good = accuracy(y_true, y_pred_good)
    
    print(f"Dataset: 90% class 0, 10% class 1")
    print(f"Naive classifier (always 0): {acc_naive:.3f}")
    print(f"Better classifier:           {acc_good:.3f}")
    print("Note: Naive classifier has high accuracy but is useless!")
    print("This shows why accuracy can be misleading with imbalanced data.")
    
    print("\nKey Insights:")
    print("✅ Use accuracy when classes are balanced")
    print("✅ Easy to understand and interpret")
    print("❌ Can be misleading with imbalanced data")
    print("❌ Doesn't tell you about specific error types")
    print("💡 Consider precision, recall, and F1-score for better evaluation")
