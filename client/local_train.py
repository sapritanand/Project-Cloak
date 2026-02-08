"""
Client-Side Training Module.
Each device trains locally and sends only gradients (with DP noise).
"""
import json
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from client.models.search_model import SearchRankingModel, extract_features
from client.differential_privacy import DifferentialPrivacy


class LocalClient:
    """
    Represents a single device/client in federated learning.
    
    Key Features:
    - Trains on local data only
    - Never sends raw data to server
    - Applies differential privacy to gradients
    """
    
    def __init__(self, client_id, data_path, privacy_level='medium'):
        """
        Args:
            client_id: Unique identifier for this client
            data_path: Path to local data file
            privacy_level: 'none', 'low', 'medium', 'high', or 'very_high'
        """
        self.client_id = client_id
        self.data_path = Path(data_path)
        
        # Initialize model
        self.model = SearchRankingModel()
        
        # Initialize differential privacy
        self.dp = DifferentialPrivacy.from_privacy_level(privacy_level)
        
        # Load local data
        self.load_data()
        
        # Training history
        self.history = {
            'loss': [],
            'accuracy': [],
            'privacy_metrics': []
        }
    
    def load_data(self):
        """Load and prepare local training data."""
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Extract features and labels
        X_list = []
        y_list = []
        
        for sample in raw_data:
            features = extract_features(sample['query'], sample['document'])
            X_list.append(features)
            y_list.append(sample['click'])
        
        self.X = np.array(X_list)
        self.y = np.array(y_list).reshape(-1, 1)
        
        # Split into train/val (80/20)
        split_idx = int(0.8 * len(self.X))
        self.X_train = self.X[:split_idx]
        self.y_train = self.y[:split_idx]
        self.X_val = self.X[split_idx:]
        self.y_val = self.y[split_idx:]
        
        print(f"Client {self.client_id}: Loaded {len(self.X)} samples "
              f"(train={len(self.X_train)}, val={len(self.X_val)})")
    
    def set_global_model(self, global_params):
        """
        Receive and set global model parameters from server.
        
        Args:
            global_params: Dictionary of model parameters
        """
        self.model.set_parameters(global_params)
    
    def train_local_epoch(self, batch_size=32):
        """
        Train model for one epoch on local data.
        
        Args:
            batch_size: Mini-batch size
        
        Returns:
            Average loss for the epoch
        """
        n_samples = len(self.X_train)
        indices = np.random.permutation(n_samples)
        
        epoch_losses = []
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:min(i + batch_size, n_samples)]
            X_batch = self.X_train[batch_idx]
            y_batch = self.y_train[batch_idx]
            
            # Training step
            loss, gradients = self.model.train_step(X_batch, y_batch)
            
            # Update local model
            self.model.update_weights(gradients)
            
            epoch_losses.append(loss)
        
        return np.mean(epoch_losses)
    
    def compute_gradients(self):
        """
        Compute gradients on local data.
        Used for federated averaging.
        
        Returns:
            Dictionary of gradients
        """
        # Compute on full training set
        _, gradients = self.model.train_step(self.X_train, self.y_train)
        return gradients
    
    def train_and_get_update(self, num_epochs=2):
        """
        Main training loop for federated learning.
        
        Steps:
        1. Train locally for num_epochs
        2. Compute final gradients
        3. Apply differential privacy
        4. Return privatized gradients
        
        Args:
            num_epochs: Number of local training epochs
        
        Returns:
            Privatized gradients
        """
        print(f"\nClient {self.client_id}: Starting local training...")
        
        # Local training
        for epoch in range(num_epochs):
            loss = self.train_local_epoch()
            self.history['loss'].append(loss)
            
            # Validate
            metrics = self.model.evaluate(self.X_val, self.y_val)
            self.history['accuracy'].append(metrics['accuracy'])
            
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"loss={loss:.4f}, val_acc={metrics['accuracy']:.2%}")
        
        # Compute gradients
        gradients = self.compute_gradients()
        
        # Apply differential privacy
        private_gradients = self.dp.privatize_gradients(gradients)
        
        # Log privacy metrics
        privacy_metrics = self.dp.get_privacy_metrics()
        self.history['privacy_metrics'].append(privacy_metrics)
        
        print(f"  ✓ Privacy applied: σ={self.dp.sigma}, "
              f"noise_mag={privacy_metrics['avg_noise_magnitude']:.4f}")
        
        return private_gradients
    
    def evaluate(self):
        """
        Evaluate current model on validation set.
        
        Returns:
            Dictionary with accuracy and loss
        """
        return self.model.evaluate(self.X_val, self.y_val)
    
    def get_training_summary(self):
        """Get summary of training performance."""
        if not self.history['accuracy']:
            return None
        
        return {
            'client_id': self.client_id,
            'final_accuracy': self.history['accuracy'][-1],
            'final_loss': self.history['loss'][-1],
            'data_size': len(self.X),
            'privacy_sigma': self.dp.sigma
        }


def test_local_client():
    """Test LocalClient with dummy data."""
    print("Testing LocalClient...")
    
    # Create dummy data
    dummy_data = [
        {"query": "test query", "document": "test document", "click": 1}
        for _ in range(50)
    ]
    
    data_path = Path("test_data.json")
    with open(data_path, 'w') as f:
        json.dump(dummy_data, f)
    
    # Create client
    client = LocalClient(
        client_id=0,
        data_path=data_path,
        privacy_level='medium'
    )
    
    # Train and get update
    gradients = client.train_and_get_update(num_epochs=2)
    
    # Evaluate
    metrics = client.evaluate()
    print(f"\nFinal metrics: {metrics}")
    
    # Cleanup
    data_path.unlink()
    
    print("\n✓ LocalClient test passed!")


if __name__ == "__main__":
    test_local_client()
