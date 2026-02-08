"""
Federated Learning Simulation.
Orchestrates training across multiple clients with privacy controls.
"""
import numpy as np
from pathlib import Path
import sys
import json
import argparse

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=""):
        print(f"{desc}...")
        return iterable

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from client.local_train import LocalClient
from server.federated_server import FederatedServer


class FederatedSimulation:
    """
    Simulation environment for federated learning.
    
    This class orchestrates:
    - Multiple clients with local data
    - Federated training rounds
    - Privacy-accuracy trade-off analysis
    """
    
    def __init__(self, num_clients=5, privacy_level='medium', data_dir='client/local_data'):
        """
        Args:
            num_clients: Number of clients to simulate
            privacy_level: Privacy level for all clients
            data_dir: Directory containing client data files
        """
        self.num_clients = num_clients
        self.privacy_level = privacy_level
        self.data_dir = Path(data_dir)
        
        # Initialize server
        self.server = FederatedServer()
        
        # Initialize clients
        self.clients = self._initialize_clients()
        
        print(f"\n{'='*60}")
        print(f"FEDERATED LEARNING SIMULATION")
        print(f"{'='*60}")
        print(f"Clients: {num_clients}")
        print(f"Privacy Level: {privacy_level}")
        print(f"{'='*60}\n")
    
    def _initialize_clients(self):
        """Initialize client instances."""
        clients = []
        
        for i in range(self.num_clients):
            data_file = self.data_dir / f"device_{i}_data.json"
            
            if not data_file.exists():
                raise FileNotFoundError(
                    f"Data file not found: {data_file}\n"
                    f"Run 'python generate_data.py' first."
                )
            
            client = LocalClient(
                client_id=i,
                data_path=data_file,
                privacy_level=self.privacy_level
            )
            clients.append(client)
        
        return clients
    
    def run_round(self, num_local_epochs=2, participation_rate=1.0):
        """
        Execute one federated learning round.
        
        Args:
            num_local_epochs: Number of local training epochs per client
            participation_rate: Fraction of clients participating (for partial updates)
        
        Returns:
            Round metrics
        """
        # Select participating clients
        n_participants = max(1, int(self.num_clients * participation_rate))
        participating = np.random.choice(
            self.clients,
            size=n_participants,
            replace=False
        )
        
        # Distribute global model to clients
        global_params = self.server.get_global_model()
        for client in participating:
            client.set_global_model(global_params)
        
        # Collect updates from clients
        client_gradients = []
        client_weights = []
        client_metrics = []
        
        for client in participating:
            # Local training + DP noise
            gradients = client.train_and_get_update(num_epochs=num_local_epochs)
            client_gradients.append(gradients)
            
            # Weight by dataset size (for FedAvg)
            client_weights.append(len(client.X_train))
            
            # Collect metrics
            metrics = client.evaluate()
            client_metrics.append(metrics)
        
        # Aggregate and update global model
        updated_model = self.server.federated_round(
            client_gradients,
            client_weights=client_weights,
            client_metrics=client_metrics
        )
        
        # Distribute updated model back to all clients
        for client in self.clients:
            client.set_global_model(updated_model)
        
        # Return round summary
        return {
            'participating_clients': n_participants,
            'avg_accuracy': np.mean([m['accuracy'] for m in client_metrics]),
            'avg_loss': np.mean([m['loss'] for m in client_metrics])
        }
    
    def train(self, num_rounds=10, num_local_epochs=2, participation_rate=1.0):
        """
        Run full federated training.
        
        Args:
            num_rounds: Number of federated rounds
            num_local_epochs: Local epochs per round
            participation_rate: Fraction of clients per round
        
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"STARTING FEDERATED TRAINING")
        print(f"{'='*60}")
        print(f"Total rounds: {num_rounds}")
        print(f"Local epochs per round: {num_local_epochs}")
        print(f"Client participation rate: {participation_rate:.0%}")
        print(f"{'='*60}\n")
        
        for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
            round_metrics = self.run_round(
                num_local_epochs=num_local_epochs,
                participation_rate=participation_rate
            )
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        
        # Get final summary
        summary = self.server.get_training_summary()
        
        print(f"\nFinal Results:")
        print(f"  Total Rounds: {summary['total_rounds']}")
        print(f"  Final Accuracy: {summary['final_accuracy']:.2%}")
        print(f"  Best Accuracy: {summary['best_accuracy']:.2%}")
        print(f"  Best Round: {summary['convergence_round']}")
        
        # Get privacy summary from first client
        privacy_summary = self.clients[0].dp.get_privacy_summary(num_rounds)
        print(f"\nPrivacy Guarantee:")
        print(f"  Noise Level (σ): {privacy_summary['sigma']}")
        print(f"  Epsilon (ε): {privacy_summary['epsilon']}")
        print(f"  Privacy Level: {privacy_summary['privacy_level']}")
        print(f"  {privacy_summary['explanation']}")
        
        print(f"{'='*60}\n")
        
        return self.server.history
    
    def evaluate_all_clients(self):
        """Evaluate global model on all clients."""
        print(f"\n{'='*60}")
        print(f"EVALUATING GLOBAL MODEL ON ALL CLIENTS")
        print(f"{'='*60}\n")
        
        results = []
        for client in self.clients:
            metrics = client.evaluate()
            results.append({
                'client_id': client.client_id,
                'accuracy': metrics['accuracy'],
                'loss': metrics['loss'],
                'data_size': len(client.X)
            })
            print(f"Client {client.client_id}: "
                  f"acc={metrics['accuracy']:.2%}, "
                  f"loss={metrics['loss']:.4f}, "
                  f"data={len(client.X)}")
        
        avg_acc = np.mean([r['accuracy'] for r in results])
        print(f"\nAverage Accuracy: {avg_acc:.2%}")
        print(f"{'='*60}\n")
        
        return results
    
    def save_results(self, output_dir='results'):
        """Save simulation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save server checkpoint
        self.server.save_checkpoint(output_path / 'global_model.pkl')
        
        # Save training history
        self.server.export_history(output_path / 'training_history.json')
        
        # Save client summaries
        client_summaries = [c.get_training_summary() for c in self.clients]
        with open(output_path / 'client_summaries.json', 'w') as f:
            json.dump(client_summaries, f, indent=2)
        
        print(f"✓ Results saved to {output_path}/")


def privacy_comparison_experiment(num_rounds=10):
    """
    Run experiments with different privacy levels.
    Demonstrates privacy-accuracy trade-off.
    """
    privacy_levels = ['none', 'low', 'medium', 'high']
    results = {}
    
    print("\n" + "="*70)
    print("PRIVACY-ACCURACY TRADE-OFF EXPERIMENT")
    print("="*70)
    
    for level in privacy_levels:
        print(f"\n{'*'*70}")
        print(f"Testing Privacy Level: {level.upper()}")
        print(f"{'*'*70}\n")
        
        # Create simulation
        sim = FederatedSimulation(
            num_clients=5,
            privacy_level=level,
            data_dir='client/local_data'
        )
        
        # Train
        history = sim.train(num_rounds=num_rounds, num_local_epochs=2)
        
        # Get results
        summary = sim.server.get_training_summary()
        privacy_summary = sim.clients[0].dp.get_privacy_summary(num_rounds)
        
        results[level] = {
            'final_accuracy': summary['final_accuracy'],
            'best_accuracy': summary['best_accuracy'],
            'sigma': privacy_summary['sigma'],
            'epsilon': privacy_summary['epsilon'],
            'privacy_level': privacy_summary['privacy_level']
        }
        
        # Save results
        sim.save_results(f'results/privacy_{level}')
    
    # Print comparison
    print("\n" + "="*70)
    print("PRIVACY-ACCURACY TRADE-OFF RESULTS")
    print("="*70)
    print(f"{'Privacy Level':<15} {'Sigma':<8} {'Epsilon':<10} {'Accuracy':<10}")
    print("-"*70)
    
    for level, metrics in results.items():
        print(f"{level:<15} {metrics['sigma']:<8} "
              f"{str(metrics['epsilon']):<10} "
              f"{metrics['final_accuracy']:.2%}")
    
    print("="*70)
    print("\nKey Insight: Higher privacy (larger σ) → Lower accuracy")
    print("This demonstrates the fundamental privacy-utility trade-off!")
    print("="*70 + "\n")
    
    # Save comparison
    with open('results/privacy_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'comparison'],
                       help='Run mode: single simulation or privacy comparison')
    parser.add_argument('--privacy', type=str, default='medium',
                       choices=['none', 'low', 'medium', 'high', 'very_high'],
                       help='Privacy level for single mode')
    parser.add_argument('--rounds', type=int, default=10,
                       help='Number of federated rounds')
    parser.add_argument('--clients', type=int, default=5,
                       help='Number of clients')
    
    args = parser.parse_args()
    
    if args.mode == 'comparison':
        # Run privacy comparison experiment
        privacy_comparison_experiment(num_rounds=args.rounds)
    else:
        # Run single simulation
        sim = FederatedSimulation(
            num_clients=args.clients,
            privacy_level=args.privacy,
            data_dir='client/local_data'
        )
        
        # Train
        history = sim.train(num_rounds=args.rounds, num_local_epochs=2)
        
        # Evaluate
        sim.evaluate_all_clients()
        
        # Save results
        sim.save_results(f'results/privacy_{args.privacy}')


if __name__ == "__main__":
    main()
