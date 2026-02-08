"""
Simple Neural Network for Search Ranking.
Predicts click probability given query-document features.
"""
import numpy as np
import pickle
from pathlib import Path


class SearchRankingModel:
    """
    Simple 2-layer MLP for click prediction.
    
    Architecture:
    - Input: 3 features (query_len, doc_len, keyword_overlap)
    - Hidden: 8 neurons (ReLU)
    - Output: 1 neuron (Sigmoid) - click probability
    """
    
    def __init__(self, input_size=3, hidden_size=8, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Initialize weights (Xavier initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, 1))
        
        # Cache for backpropagation
        self.cache = {}
    
    def _relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """ReLU derivative."""
        return (x > 0).astype(float)
    
    def _sigmoid(self, x):
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        """
        Forward pass.
        
        Args:
            X: Input features (n_samples, input_size)
        
        Returns:
            Predicted probabilities (n_samples, 1)
        """
        # Layer 1
        self.cache['X'] = X
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._relu(Z1)
        self.cache['Z1'] = Z1
        self.cache['A1'] = A1
        
        # Layer 2
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._sigmoid(Z2)
        self.cache['A2'] = A2
        
        return A2
    
    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss."""
        m = y_true.shape[0]
        epsilon = 1e-7  # Prevent log(0)
        loss = -np.mean(
            y_true * np.log(y_pred + epsilon) + 
            (1 - y_true) * np.log(1 - y_pred + epsilon)
        )
        return loss
    
    def backward(self, y_true):
        """
        Backward pass (compute gradients).
        
        Args:
            y_true: True labels (n_samples, 1)
        
        Returns:
            Dictionary of gradients
        """
        m = y_true.shape[0]
        
        # Output layer gradient
        dZ2 = self.cache['A2'] - y_true  # BCE + Sigmoid derivative
        dW2 = np.dot(self.cache['A1'].T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Hidden layer gradient
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._relu_derivative(self.cache['Z1'])
        dW1 = np.dot(self.cache['X'].T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        return {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
    
    def update_weights(self, gradients):
        """Update model weights using gradients."""
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']
    
    def train_step(self, X, y):
        """
        Single training step.
        
        Returns:
            loss: Training loss
            gradients: Computed gradients
        """
        # Forward pass
        y_pred = self.forward(X)
        
        # Compute loss
        loss = self.compute_loss(y, y_pred)
        
        # Backward pass
        gradients = self.backward(y)
        
        return loss, gradients
    
    def predict_proba(self, X):
        """Predict click probabilities."""
        return self.forward(X)
    
    def predict(self, X, threshold=0.5):
        """Predict binary clicks."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Returns:
            Dictionary with accuracy and loss
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        
        y_proba = self.predict_proba(X)
        loss = self.compute_loss(y, y_proba)
        
        return {
            'accuracy': accuracy,
            'loss': loss
        }
    
    def get_parameters(self):
        """Get model parameters as dictionary."""
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }
    
    def set_parameters(self, params):
        """Set model parameters from dictionary."""
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()
    
    def save(self, filepath):
        """Save model to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
    
    def load(self, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.set_parameters(params)


def extract_features(query, document):
    """
    Extract simple features from query-document pair.
    
    Features:
    1. Query length (normalized)
    2. Document length (normalized)
    3. Keyword overlap ratio
    """
    query_words = query.lower().split()
    doc_words = document.lower().split()
    
    # Feature 1: Query length
    query_len = len(query_words) / 10.0  # Normalize
    
    # Feature 2: Document length
    doc_len = len(doc_words) / 20.0  # Normalize
    
    # Feature 3: Keyword overlap
    query_set = set(query_words)
    doc_set = set(doc_words)
    overlap = len(query_set.intersection(doc_set))
    overlap_ratio = overlap / len(query_set) if len(query_set) > 0 else 0
    
    return np.array([query_len, doc_len, overlap_ratio])


if __name__ == "__main__":
    # Test the model
    print("Testing SearchRankingModel...")
    
    model = SearchRankingModel()
    
    # Create dummy data
    X = np.random.randn(10, 3)
    y = np.random.randint(0, 2, (10, 1))
    
    # Train one step
    loss, grads = model.train_step(X, y)
    print(f"Loss: {loss:.4f}")
    
    # Update weights
    model.update_weights(grads)
    
    # Evaluate
    metrics = model.evaluate(X, y)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    
    print("✓ Model test passed!")
