import numpy as np
from typing import Tuple, Dict, Any
from sklearn.datasets import make_classification, make_regression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score

class TaskGenerator:
    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)
        
    def generate_classification_task(self, n_samples: int = 1000, n_features: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a classification task with synthetic data."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=5,
            random_state=self.rng
        )
        return X, y
    
    def generate_regression_task(self, n_samples: int = 1000, n_features: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a regression task with synthetic data."""
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            noise=0.1,
            random_state=self.rng
        )
        return X, y
    
    def generate_clustering_task(self, n_samples: int = 1000, n_features: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a clustering task with synthetic data."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=self.rng
        )
        return X, y

class TaskEvaluator:
    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate classification performance."""
        return {
            'accuracy': accuracy_score(y_true, y_pred)
        }
    
    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate regression performance."""
        return {
            'mse': mean_squared_error(y_true, y_pred)
        }
    
    @staticmethod
    def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering performance."""
        return {
            'silhouette': silhouette_score(X, labels)
        } 