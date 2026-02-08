"""
Visualization tools for federated learning results.
Creates publication-quality plots for privacy-accuracy trade-offs.
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def plot_training_history(history_path, save_path=None):
    """
    Plot training curves (accuracy and loss over rounds).
    
    Args:
        history_path: Path to training_history.json
        save_path: Optional path to save figure
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = history['rounds']
    accuracy = [a * 100 for a in history['avg_accuracy']]  # Convert to percentage
    loss = history['avg_loss']
    
    # Plot accuracy
    ax1.plot(rounds, accuracy, marker='o', linewidth=2, markersize=6, color='#2ecc71')
    ax1.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Over Rounds', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Plot loss
    ax2.plot(rounds, loss, marker='s', linewidth=2, markersize=6, color='#e74c3c')
    ax2.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Training Loss Over Rounds', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    return fig


def plot_privacy_accuracy_tradeoff(comparison_path, save_path=None):
    """
    Plot privacy-accuracy trade-off across different privacy levels.
    
    Args:
        comparison_path: Path to privacy_comparison.json
        save_path: Optional path to save figure
    """
    with open(comparison_path, 'r') as f:
        results = json.load(f)
    
    # Extract data
    privacy_levels = list(results.keys())
    accuracies = [results[level]['final_accuracy'] * 100 for level in privacy_levels]
    sigmas = [results[level]['sigma'] for level in privacy_levels]
    epsilons = [results[level]['epsilon'] for level in privacy_levels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy vs Sigma
    colors = ['#27ae60', '#f39c12', '#e67e22', '#c0392b']
    ax1.bar(privacy_levels, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Privacy Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Privacy Level vs Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add sigma values on bars
    for i, (level, acc, sigma) in enumerate(zip(privacy_levels, accuracies, sigmas)):
        ax1.text(i, acc + 2, f'σ={sigma}', ha='center', fontweight='bold', fontsize=10)
    
    # Plot 2: Privacy Budget (Epsilon)
    eps_display = [float(e) if e != '∞' else 100 for e in epsilons]
    ax2.plot(privacy_levels, eps_display, marker='o', linewidth=3, 
             markersize=10, color='#9b59b6')
    ax2.set_xlabel('Privacy Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Privacy Budget (ε)', fontsize=12, fontweight='bold')
    ax2.set_title('Privacy Budget Across Levels', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Add annotation
    ax2.annotate('Lower ε = Stronger Privacy', 
                xy=(0.5, 0.95), xycoords='axes fraction',
                fontsize=11, ha='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    return fig


def plot_client_performance(summaries_path, save_path=None):
    """
    Plot performance across different clients.
    
    Args:
        summaries_path: Path to client_summaries.json
        save_path: Optional path to save figure
    """
    with open(summaries_path, 'r') as f:
        summaries = json.load(f)
    
    client_ids = [s['client_id'] for s in summaries]
    accuracies = [s['final_accuracy'] * 100 for s in summaries]
    data_sizes = [s['data_size'] for s in summaries]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Client Accuracy
    ax1.bar(client_ids, accuracies, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Client ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Client Performance', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Data Distribution
    ax2.bar(client_ids, data_sizes, color='#1abc9c', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Client ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Data Distribution Across Clients', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    return fig


def create_comprehensive_report(results_dir='results'):
    """
    Generate all visualizations for a results directory.
    
    Args:
        results_dir: Directory containing result files
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"GENERATING VISUALIZATION REPORT")
    print(f"{'='*60}\n")
    
    # Create visualization directory
    viz_dir = results_path / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Training history (if exists)
    history_file = results_path / 'training_history.json'
    if history_file.exists():
        print("Creating training history plot...")
        plot_training_history(
            history_file,
            save_path=viz_dir / 'training_history.png'
        )
        plt.close()
    
    # 2. Client performance (if exists)
    summaries_file = results_path / 'client_summaries.json'
    if summaries_file.exists():
        print("Creating client performance plot...")
        plot_client_performance(
            summaries_file,
            save_path=viz_dir / 'client_performance.png'
        )
        plt.close()
    
    # 3. Privacy comparison (if exists)
    comparison_file = Path('results/privacy_comparison.json')
    if comparison_file.exists():
        print("Creating privacy-accuracy trade-off plot...")
        plot_privacy_accuracy_tradeoff(
            comparison_file,
            save_path=Path('results/visualizations/privacy_tradeoff.png')
        )
        plt.close()
    
    print(f"\n✓ All visualizations saved to {viz_dir}/")
    print(f"{'='*60}\n")


def create_summary_dashboard():
    """
    Create a comprehensive dashboard combining all experiments.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Load all data
    comparison_file = Path('results/privacy_comparison.json')
    
    if not comparison_file.exists():
        print("Error: Run privacy comparison experiment first!")
        return
    
    with open(comparison_file, 'r') as f:
        comparison = json.load(f)
    
    # Extract data
    privacy_levels = list(comparison.keys())
    accuracies = [comparison[level]['final_accuracy'] * 100 for level in privacy_levels]
    sigmas = [comparison[level]['sigma'] for level in privacy_levels]
    epsilons = [comparison[level]['epsilon'] for level in privacy_levels]
    
    # Plot 1: Accuracy comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#27ae60', '#f39c12', '#e67e22', '#c0392b']
    bars = ax1.bar(privacy_levels, accuracies, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Model Accuracy by Privacy Level', fontweight='bold', fontsize=13)
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=10)
    
    # Plot 2: Sigma values (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(privacy_levels, sigmas, marker='o', linewidth=3, 
             markersize=12, color='#9b59b6')
    ax2.set_ylabel('Noise Scale (σ)', fontweight='bold')
    ax2.set_title('Privacy Noise by Level', fontweight='bold', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Privacy-Accuracy Scatter (middle)
    ax3 = fig.add_subplot(gs[1, :])
    scatter = ax3.scatter(sigmas, accuracies, s=300, c=colors, alpha=0.8, 
                         edgecolors='black', linewidths=2)
    
    # Add labels
    for i, (sigma, acc, level) in enumerate(zip(sigmas, accuracies, privacy_levels)):
        ax3.annotate(level.upper(), (sigma, acc), 
                    xytext=(5, 5), textcoords='offset points',
                    fontweight='bold', fontsize=11)
    
    ax3.set_xlabel('Noise Scale (σ)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax3.set_title('Privacy-Accuracy Trade-off', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])
    
    # Add trend line
    z = np.polyfit(sigmas, accuracies, 2)
    p = np.poly1d(z)
    x_line = np.linspace(min(sigmas), max(sigmas), 100)
    ax3.plot(x_line, p(x_line), "--", color='gray', linewidth=2, alpha=0.6, label='Trend')
    ax3.legend()
    
    # Plot 4: Key Insights (bottom)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Calculate insights
    accuracy_drop = accuracies[0] - accuracies[-1]
    
    insights_text = f"""
    KEY FINDINGS:
    
    • Privacy Protection: Higher σ values provide stronger privacy guarantees
    • Accuracy Impact: Moving from 'none' to 'high' privacy reduces accuracy by {accuracy_drop:.1f}%
    • Recommended Setting: 'medium' privacy (σ=0.5) balances privacy and utility
    • Trade-off: This demonstrates the fundamental privacy-utility trade-off in ML
    
    WHY THIS MATTERS:
    Federated Learning + Differential Privacy enables AI models to learn from distributed 
    data while mathematically guaranteeing that individual data points remain private.
    """
    
    ax4.text(0.1, 0.5, insights_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            family='monospace')
    
    plt.suptitle('Privacy-Preserving Search: Comprehensive Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    save_path = Path('results/visualizations/comprehensive_dashboard.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comprehensive dashboard saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Generate all reports
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations')
    parser.add_argument('--mode', type=str, default='dashboard',
                       choices=['report', 'dashboard', 'both'],
                       help='Visualization mode')
    
    args = parser.parse_args()
    
    if args.mode in ['report', 'both']:
        create_comprehensive_report('results')
    
    if args.mode in ['dashboard', 'both']:
        create_summary_dashboard()
        plt.show()
