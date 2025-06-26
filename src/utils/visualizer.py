"""
Visualization utilities for machine learning models.

This module provides plotting and visualization functions to help
understand model performance, data distributions, and training progress.
"""
import numpy as np
import matplotlib.pyplot as plt

class ModelVisualizer:
    """
    Visualization utilities for machine learning models.
    """
    
    def __init__(self, figsize=(10, 6)):
        """
        Initialize visualizer.
        
        Args:
            figsize (tuple): Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_regression_results(self, y_true, y_pred, title="Regression Results"):
        """
        Plot regression results with actual vs predicted values.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            title (str): Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot: Actual vs Predicted
        ax1.scatter(y_true, y_pred, alpha=0.6, color='blue')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Actual vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, losses, title="Training History"):
        """
        Plot training loss over epochs.
        
        Args:
            losses (list): List of loss values per epoch
            title (str): Plot title
        """
        plt.figure(figsize=self.figsize)
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_classification_results(self, y_true, y_pred, class_names=None):
        """
        Plot classification results with confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (list): Names of classes
        """
        from ..metrics.accuracy import Accuracy
        
        # Calculate confusion matrix
        classes = np.unique(y_true)
        n_classes = len(classes)
        confusion_matrix = np.zeros((n_classes, n_classes))
        
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                confusion_matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        
        # Plot confusion matrix
        plt.figure(figsize=self.figsize)
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        if class_names is None:
            class_names = [f'Class {i}' for i in classes]
        
        tick_marks = np.arange(n_classes)
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = confusion_matrix.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                plt.text(j, i, int(confusion_matrix[i, j]),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # Calculate and print accuracy
        accuracy = Accuracy()
        acc_score = accuracy.calculate(y_true, y_pred)
        print(f"Accuracy: {acc_score:.4f}")
    
    def plot_data_distribution(self, X, y=None, feature_names=None):
        """
        Plot data distribution for features.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray, optional): Target labels for coloring
            feature_names (list): Names of features
        """
        n_features = X.shape[1]
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(n_features):
            ax = axes[i]
            
            if y is not None:
                # Color by target variable
                unique_labels = np.unique(y)
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
                
                for label, color in zip(unique_labels, colors):
                    mask = y == label
                    ax.hist(X[mask, i], alpha=0.7, color=color, label=f'Class {label}', bins=20)
                ax.legend()
            else:
                ax.hist(X[:, i], alpha=0.7, bins=20)
            
            feature_name = feature_names[i] if feature_names else f'Feature {i}'
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {feature_name}')
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curve(self, train_losses, val_losses=None, title="Learning Curve"):
        """
        Plot learning curve with training and validation losses.
        
        Args:
            train_losses (list): Training losses per epoch
            val_losses (list, optional): Validation losses per epoch
            title (str): Plot title
        """
        plt.figure(figsize=self.figsize)
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
        
        if val_losses is not None:
            plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_decision_boundary_2d(self, X, y, model, title="Decision Boundary"):
        """
        Plot decision boundary for 2D data.
        
        Args:
            X (np.ndarray): 2D input features
            y (np.ndarray): Target labels
            model: Trained model with predict method
            title (str): Plot title
        """
        if X.shape[1] != 2:
            raise ValueError("This function only works with 2D data")
        
        plt.figure(figsize=self.figsize)
        
        # Create a mesh
        h = 0.02  # step size
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        
        # Plot the data points
        unique_labels = np.unique(y)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = y == label
            plt.scatter(X[mask, 0], X[mask, 1], c=[color], label=f'Class {label}', 
                       edgecolors='black', s=50)
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.show()
