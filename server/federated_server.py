"""
Federated Learning Server.
Aggregates client updates and maintains global model.
"""
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime


class FederatedServer:
    """
    Centralized server for federated learning.
    
    Responsibilities:
    - Maintain global model
    - Aggregate client gradients (FedAvg)
    - Distribute updated model to clients
    """
    
    def __init__(self, model_config=None):
        """
        Args:
            model_config: Initial model configuration (optional)
        """
        # Import here to avoid circular dependency
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from client.models.search_model import SearchRankingModel
        
        # Initialize global model
        self.global_model = SearchRankingModel()
        
        # Training history
        self.history = {
            'rounds': [],
            'avg_accuracy': [],
            'avg_loss': [],
            'num_clients': []
        }
        
        # Current round
        self.current_round = 0
    
    def get_global_model(self):
        """
        Get current global model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.global_model.get_parameters()
    
    def aggregate_gradients(self, client_gradients, weights=None):
        """
        Aggregate gradients from multiple clients using FedAvg.
        
        Args:
            client_gradients: List of gradient dictionaries from clients
            weights: Optional weights for each client (e.g., by data size)
                    If None, simple averaging is used
        
        Returns:
            Aggregated gradients
        """
        n_clients = len(client_gradients)
        
        if n_clients == 0:
            raise ValueError("No client gradients to aggregate")
        
        # Use uniform weights if not provided
        if weights is None:
            weights = np.ones(n_clients) / n_clients
        else:
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
        
        # Initialize aggregated gradients
        aggregated = {}
        
        # Get gradient keys from first client
        grad_keys = client_gradients[0].keys()
        
        # Weighted average of gradients
        for key in grad_keys:
            aggregated[key] = sum(
                weight * client_grads[key]
                for weight, client_grads in zip(weights, client_gradients)
            )
        
        return aggregated
    
    def update_global_model(self, aggregated_gradients):
        """
        Update global model using aggregated gradients.
        
        Args:
            aggregated_gradients: Aggregated gradient dictionary
        """
        self.global_model.update_weights(aggregated_gradients)
    
    def federated_round(self, client_gradients, client_weights=None, 
                       client_metrics=None):
        """
        Execute one round of federated learning.
        
        Args:
            client_gradients: List of gradient dicts from clients
            client_weights: Optional weights for each client
            client_metrics: Optional metrics from each client (for logging)
        
        Returns:
            Updated global model parameters
        """
        self.current_round += 1
        
        print(f"\n{'='*60}")
        print(f"FEDERATED ROUND {self.current_round}")
        print(f"{'='*60}")
        print(f"Participating clients: {len(client_gradients)}")
        
        # Aggregate gradients
        aggregated = self.aggregate_gradients(client_gradients, client_weights)
        
        # Update global model
        self.update_global_model(aggregated)
        
        # Log metrics
        if client_metrics:
            avg_acc = np.mean([m['accuracy'] for m in client_metrics])
            avg_loss = np.mean([m['loss'] for m in client_metrics])
            
            self.history['rounds'].append(self.current_round)
            self.history['avg_accuracy'].append(avg_acc)
            self.history['avg_loss'].append(avg_loss)
            self.history['num_clients'].append(len(client_gradients))
            
            print(f"Average client accuracy: {avg_acc:.2%}")
            print(f"Average client loss: {avg_loss:.4f}")
        
        print(f"{'='*60}\n")
        
        return self.get_global_model()
    
    def save_checkpoint(self, filepath):
        """
        Save server state and global model.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'global_model': self.global_model.get_parameters(),
            'history': self.history,
            'current_round': self.current_round,
            'timestamp': datetime.now().isoformat()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"✓ Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Load server state and global model.
        
        Args:
            filepath: Path to checkpoint file
        """
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.global_model.set_parameters(checkpoint['global_model'])
        self.history = checkpoint['history']
        self.current_round = checkpoint['current_round']
        
        print(f"✓ Checkpoint loaded from {filepath}")
        print(f"  Round: {self.current_round}")
    
    def get_training_summary(self):
        """
        Get summary of federated training.
        
        Returns:
            Dictionary with training statistics
        """
        if not self.history['rounds']:
            return {
                'total_rounds': 0,
                'final_accuracy': None,
                'final_loss': None
            }
        
        return {
            'total_rounds': self.current_round,
            'final_accuracy': self.history['avg_accuracy'][-1],
            'final_loss': self.history['avg_loss'][-1],
            'best_accuracy': max(self.history['avg_accuracy']),
            'convergence_round': np.argmax(self.history['avg_accuracy']) + 1
        }
    
    def export_history(self, filepath):
        """
        Export training history to JSON.
        
        Args:
            filepath: Path to save history
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON
        history_export = {
            'rounds': [int(r) for r in self.history['rounds']],
            'avg_accuracy': [float(a) for a in self.history['avg_accuracy']],
            'avg_loss': [float(l) for l in self.history['avg_loss']],
            'num_clients': [int(n) for n in self.history['num_clients']],
            'summary': {
                'total_rounds': int(self.get_training_summary()['total_rounds']),
                'final_accuracy': float(self.get_training_summary()['final_accuracy']) if self.get_training_summary()['final_accuracy'] is not None else None,
                'final_loss': float(self.get_training_summary()['final_loss']) if self.get_training_summary()['final_loss'] is not None else None,
                'best_accuracy': float(self.get_training_summary()['best_accuracy']) if 'best_accuracy' in self.get_training_summary() else None,
                'convergence_round': int(self.get_training_summary()['convergence_round']) if 'convergence_round' in self.get_training_summary() else None
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_export, f, indent=2)
        
        print(f"✓ Training history exported to {filepath}")


def test_federated_server():
    """Test FederatedServer with dummy gradients."""
    print("Testing FederatedServer...")
    
    server = FederatedServer()
    
    # Simulate 3 clients with dummy gradients
    client_gradients = [
        {
            'dW1': np.random.randn(3, 8) * 0.01,
            'db1': np.random.randn(1, 8) * 0.01,
            'dW2': np.random.randn(8, 1) * 0.01,
            'db2': np.random.randn(1, 1) * 0.01
        }
        for _ in range(3)
    ]
    
    client_metrics = [
        {'accuracy': 0.7 + np.random.rand() * 0.1, 'loss': 0.5}
        for _ in range(3)
    ]
    
    # Run one federated round
    updated_model = server.federated_round(
        client_gradients,
        client_metrics=client_metrics
    )
    
    print(f"Updated model keys: {list(updated_model.keys())}")
    
    # Get summary
    summary = server.get_training_summary()
    print(f"\nTraining summary: {summary}")
    
    print("\n✓ FederatedServer test passed!")


if __name__ == "__main__":
    test_federated_server()
