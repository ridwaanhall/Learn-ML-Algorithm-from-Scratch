"""
Machine Learning from Scratch - Main Demo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import our ML framework
from models.linear_regression import LinearRegression, RidgeRegression
from models.logistic_regression import LogisticRegression
from models.knn import KNNClassifier, KNNRegressor
from models.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from models.kmeans import KMeans
from models.pca import PCA

from preprocessing.scaler import StandardScaler, MinMaxScaler
from preprocessing.encoder import LabelEncoder, OneHotEncoder
from preprocessing.split import TrainTestSplit

from metrics.classification import ClassificationMetrics
from metrics.regression import RegressionMetrics

from utils.matrix import DataUtils

import warnings
warnings.filterwarnings('ignore')


def load_iris_data():
    """Load and prepare Iris dataset"""
    data_path = Path("data/iris.csv")
    df = pd.read_csv(data_path)
    
    # Features and target
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y = df['species'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded, label_encoder.classes_


def load_housing_data():
    """Load and prepare Housing dataset"""
    data_path = Path("data/housing.csv")
    df = pd.read_csv(data_path)
    
    # Features and target
    X = df[['area', 'bedrooms', 'age']].values
    y = df['price'].values
    
    return X, y


def demo_linear_regression():
    """Demonstrate Linear Regression"""
    print("="*60)
    print("LINEAR REGRESSION DEMO")
    print("="*60)
    
    # Load housing data
    X, y = load_housing_data()
    
    # Split data
    X_train, X_test, y_train, y_test = TrainTestSplit.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Normal equation solver
    print("\n1. Linear Regression (Normal Equation)")
    lr_normal = LinearRegression(solver='normal_equation')
    lr_normal.fit(X_train_scaled, y_train)
    
    y_pred_normal = lr_normal.predict(X_test_scaled)
    
    print(f"R² Score: {RegressionMetrics.r2_score(y_test, y_pred_normal):.4f}")
    print(f"RMSE: {RegressionMetrics.root_mean_squared_error(y_test, y_pred_normal):.2f}")
    print(f"MAE: {RegressionMetrics.mean_absolute_error(y_test, y_pred_normal):.2f}")
    
    # SGD solver
    print("\n2. Linear Regression (SGD)")
    lr_sgd = LinearRegression(solver='sgd', learning_rate=0.01, max_iterations=1000)
    lr_sgd.fit(X_train_scaled, y_train)
    
    y_pred_sgd = lr_sgd.predict(X_test_scaled)
    
    print(f"R² Score: {RegressionMetrics.r2_score(y_test, y_pred_sgd):.4f}")
    print(f"RMSE: {RegressionMetrics.root_mean_squared_error(y_test, y_pred_sgd):.2f}")
    print(f"Convergence: {len(lr_sgd.loss_history)} iterations")
    
    # Ridge Regression
    print("\n3. Ridge Regression")
    ridge = RidgeRegression(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    
    y_pred_ridge = ridge.predict(X_test_scaled)
    
    print(f"R² Score: {RegressionMetrics.r2_score(y_test, y_pred_ridge):.4f}")
    print(f"RMSE: {RegressionMetrics.root_mean_squared_error(y_test, y_pred_ridge):.2f}")


def demo_logistic_regression():
    """Demonstrate Logistic Regression"""
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION DEMO")
    print("="*60)
    
    # Load iris data
    X, y, class_names = load_iris_data()
    
    # Split data
    X_train, X_test, y_train, y_test = TrainTestSplit.train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Binary classification (setosa vs others)
    print("\n1. Binary Classification (Setosa vs Others)")
    y_binary = (y_train == 0).astype(int)
    y_test_binary = (y_test == 0).astype(int)
    
    lr_binary = LogisticRegression(max_iter=1000, learning_rate=0.1)
    lr_binary.fit(X_train_scaled, y_binary)
    
    y_pred_binary = lr_binary.predict(X_test_scaled)
    y_proba_binary = lr_binary.predict_proba(X_test_scaled)
    
    print(f"Accuracy: {ClassificationMetrics.accuracy(y_test_binary, y_pred_binary):.4f}")
    print(f"Precision: {ClassificationMetrics.precision(y_test_binary, y_pred_binary):.4f}")
    print(f"Recall: {ClassificationMetrics.recall(y_test_binary, y_pred_binary):.4f}")
    print(f"F1-Score: {ClassificationMetrics.f1_score(y_test_binary, y_pred_binary):.4f}")
    
    # Multiclass classification
    print("\n2. Multiclass Classification (All species)")
    lr_multi = LogisticRegression(max_iter=1000, learning_rate=0.1, multi_class='ovr')
    lr_multi.fit(X_train_scaled, y_train)
    
    y_pred_multi = lr_multi.predict(X_test_scaled)
    y_proba_multi = lr_multi.predict_proba(X_test_scaled)
    
    print(f"Accuracy: {ClassificationMetrics.accuracy(y_test, y_pred_multi):.4f}")
    
    # Classification report
    report = ClassificationMetrics.classification_report(y_test, y_pred_multi)
    print(f"\nClassification Report:")
    for class_idx, class_name in enumerate(class_names):
        if f'class_{class_idx}' in report:
            metrics = report[f'class_{class_idx}']
            print(f"{class_name}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")


def demo_knn():
    """Demonstrate K-Nearest Neighbors"""
    print("\n" + "="*60)
    print("K-NEAREST NEIGHBORS DEMO")
    print("="*60)
    
    # Classification
    print("\n1. KNN Classification")
    X, y, class_names = load_iris_data()
    
    X_train, X_test, y_train, y_test = TrainTestSplit.train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features for KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different k values
    for k in [3, 5, 7]:
        knn_clf = KNNClassifier(n_neighbors=k, weights='uniform')
        knn_clf.fit(X_train_scaled, y_train)
        
        y_pred = knn_clf.predict(X_test_scaled)
        accuracy = ClassificationMetrics.accuracy(y_test, y_pred)
        
        print(f"k={k}, Accuracy: {accuracy:.4f}")
    
    # Distance-weighted KNN
    knn_weighted = KNNClassifier(n_neighbors=5, weights='distance')
    knn_weighted.fit(X_train_scaled, y_train)
    y_pred_weighted = knn_weighted.predict(X_test_scaled)
    
    print(f"\nDistance-weighted (k=5), Accuracy: {ClassificationMetrics.accuracy(y_test, y_pred_weighted):.4f}")
    
    # Regression
    print("\n2. KNN Regression")
    X_reg, y_reg = load_housing_data()
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = TrainTestSplit.train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)
    
    knn_reg = KNNRegressor(n_neighbors=5, weights='distance')
    knn_reg.fit(X_train_reg_scaled, y_train_reg)
    
    y_pred_reg = knn_reg.predict(X_test_reg_scaled)
    
    print(f"R² Score: {RegressionMetrics.r2_score(y_test_reg, y_pred_reg):.4f}")
    print(f"RMSE: {RegressionMetrics.root_mean_squared_error(y_test_reg, y_pred_reg):.2f}")


def demo_decision_tree():
    """Demonstrate Decision Trees"""
    print("\n" + "="*60)
    print("DECISION TREE DEMO")
    print("="*60)
    
    # Classification
    print("\n1. Decision Tree Classification")
    X, y, class_names = load_iris_data()
    
    X_train, X_test, y_train, y_test = TrainTestSplit.train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Gini criterion
    dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    dt_gini.fit(X_train, y_train)
    
    y_pred_gini = dt_gini.predict(X_test)
    
    print(f"Gini Criterion - Accuracy: {ClassificationMetrics.accuracy(y_test, y_pred_gini):.4f}")
    
    # Entropy criterion
    dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    dt_entropy.fit(X_train, y_train)
    
    y_pred_entropy = dt_entropy.predict(X_test)
    
    print(f"Entropy Criterion - Accuracy: {ClassificationMetrics.accuracy(y_test, y_pred_entropy):.4f}")
    
    # Regression
    print("\n2. Decision Tree Regression")
    X_reg, y_reg = load_housing_data()
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = TrainTestSplit.train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    dt_reg = DecisionTreeRegressor(criterion='squared_error', max_depth=8, random_state=42)
    dt_reg.fit(X_train_reg, y_train_reg)
    
    y_pred_reg = dt_reg.predict(X_test_reg)
    
    print(f"R² Score: {RegressionMetrics.r2_score(y_test_reg, y_pred_reg):.4f}")
    print(f"RMSE: {RegressionMetrics.root_mean_squared_error(y_test_reg, y_pred_reg):.2f}")


def demo_kmeans():
    """Demonstrate K-Means Clustering"""
    print("\n" + "="*60)
    print("K-MEANS CLUSTERING DEMO")
    print("="*60)
    
    # Load iris data (unsupervised, so we'll ignore labels for clustering)
    X, y_true, class_names = load_iris_data()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try different numbers of clusters
    print("\nTesting different numbers of clusters:")
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        print(f"k={k}, Inertia: {kmeans.inertia_:.2f}, Iterations: {kmeans.n_iter_}")
    
    # Detailed analysis for k=3 (true number of species)
    print("\nDetailed analysis for k=3:")
    kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans_3.fit_predict(X_scaled)
    
    print(f"Final inertia: {kmeans_3.inertia_:.2f}")
    print(f"Centroids shape: {kmeans_3.cluster_centers_.shape}")
    
    # Compare with true labels (though this is cheating in unsupervised learning!)
    print("\nCluster distribution:")
    for i in range(3):
        cluster_mask = cluster_labels == i
        true_labels_in_cluster = y_true[cluster_mask]
        unique, counts = np.unique(true_labels_in_cluster, return_counts=True)
        
        print(f"Cluster {i}: {len(true_labels_in_cluster)} points")
        for label, count in zip(unique, counts):
            print(f"  {class_names[label]}: {count} points")


def demo_pca():
    """Demonstrate Principal Component Analysis"""
    print("\n" + "="*60)
    print("PRINCIPAL COMPONENT ANALYSIS DEMO")
    print("="*60)
    
    # Load iris data
    X, y, class_names = load_iris_data()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA with all components
    print("\n1. PCA with all components")
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_scaled)
    
    print(f"Original shape: {X_scaled.shape}")
    print(f"Transformed shape: {X_pca_full.shape}")
    print(f"Explained variance ratio: {pca_full.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca_full.explained_variance_ratio_)}")
    
    # PCA with 2 components
    print("\n2. PCA with 2 components")
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    print(f"Transformed shape: {X_pca_2d.shape}")
    print(f"Explained variance ratio: {pca_2d.explained_variance_ratio_}")
    print(f"Total explained variance: {np.sum(pca_2d.explained_variance_ratio_):.4f}")
    
    # PCA with variance threshold
    print("\n3. PCA with 95% variance retention")
    pca_95 = PCA(n_components=0.95)
    X_pca_95 = pca_95.fit_transform(X_scaled)
    
    print(f"Number of components: {pca_95.n_components_}")
    print(f"Transformed shape: {X_pca_95.shape}")
    print(f"Explained variance ratio: {pca_95.explained_variance_ratio_}")
    
    # Reconstruction
    print("\n4. Data reconstruction")
    X_reconstructed = pca_2d.inverse_transform(X_pca_2d)
    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
    
    print(f"Reconstruction error (MSE): {reconstruction_error:.6f}")


def demo_preprocessing():
    """Demonstrate preprocessing utilities"""
    print("\n" + "="*60)
    print("PREPROCESSING DEMO")
    print("="*60)
    
    # Load data
    X, y = load_housing_data()
    
    print("1. Scaling")
    print(f"Original data range: {X.min(axis=0)} to {X.max(axis=0)}")
    
    # StandardScaler
    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(X)
    print(f"After StandardScaler: mean={X_std.mean(axis=0)}, std={X_std.std(axis=0)}")
    
    # MinMaxScaler
    minmax_scaler = MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X)
    print(f"After MinMaxScaler: min={X_minmax.min(axis=0)}, max={X_minmax.max(axis=0)}")
    
    print("\n2. Encoding")
    
    # Load iris for categorical encoding demo
    X_iris, y_iris, class_names = load_iris_data()
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(class_names)
    print(f"Original classes: {class_names}")
    print(f"Encoded classes: {y_encoded}")
    print(f"Decoded classes: {label_encoder.inverse_transform(y_encoded)}")
    
    # One-hot encoding
    categorical_data = np.array(['red', 'blue', 'green', 'red', 'blue']).reshape(-1, 1)
    onehot_encoder = OneHotEncoder()
    encoded_categorical = onehot_encoder.fit_transform(categorical_data)
    print(f"\nCategorical data: {categorical_data.flatten()}")
    print(f"One-hot encoded shape: {encoded_categorical.shape}")
    print(f"Feature names: {onehot_encoder.get_feature_names_out(['color'])}")
    
    print("\n3. Train-Test Split")
    X_train, X_test, y_train, y_test = TrainTestSplit.train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
    )
    
    print(f"Original shape: {X_iris.shape}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Check stratification
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    print(f"Train class distribution: {dict(zip(unique_train, counts_train))}")
    print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")


def main():
    """Main demonstration function"""
    print("MACHINE LEARNING FROM SCRATCH - COMPREHENSIVE DEMO")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Run all demonstrations
        demo_preprocessing()
        demo_linear_regression()
        demo_logistic_regression()
        demo_knn()
        demo_decision_tree()
        demo_kmeans()
        demo_pca()
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nFramework Components Demonstrated:")
        print("✓ Linear Regression (Normal Equation & SGD)")
        print("✓ Ridge Regression")
        print("✓ Logistic Regression (Binary & Multiclass)")
        print("✓ K-Nearest Neighbors (Classification & Regression)")
        print("✓ Decision Trees (Classification & Regression)")
        print("✓ K-Means Clustering")
        print("✓ Principal Component Analysis (PCA)")
        print("✓ Data Preprocessing (Scaling, Encoding, Splitting)")
        print("✓ Evaluation Metrics (Classification & Regression)")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
