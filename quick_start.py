#!/usr/bin/env python3
"""
Quick Start Script for Privacy-Preserving Search
Runs a complete demo with data generation, training, and visualization.
"""
import subprocess
import sys
from pathlib import Path


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"▶ {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} complete!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {description} failed!")
        print(f"  {e}")
        return False


def main():
    """Run complete demo workflow."""
    print("\n" + "="*70)
    print("  🔐 PRIVACY-PRESERVING SEARCH - QUICK START DEMO")
    print("="*70)
    print("\nThis script will:")
    print("  1. Generate synthetic search data")
    print("  2. Run federated learning experiments")
    print("  3. Generate comprehensive visualizations")
    print("\nEstimated time: 3-5 minutes")
    
    # Confirm
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    # Step 1: Generate data
    print_section("STEP 1: Generating Synthetic Data")
    if not run_command(
        "python generate_data.py",
        "Data generation"
    ):
        return
    
    # Step 2: Run privacy comparison
    print_section("STEP 2: Training Models with Different Privacy Levels")
    print("This will train 4 models (none, low, medium, high privacy)")
    print("Each model trains for 10 federated rounds\n")
    
    if not run_command(
        "python simulation/run_federated.py --mode comparison --rounds 10",
        "Privacy comparison experiment"
    ):
        return
    
    # Step 3: Generate visualizations
    print_section("STEP 3: Generating Visualizations")
    if not run_command(
        "python simulation/visualize.py --mode both",
        "Visualization generation"
    ):
        return
    
    # Summary
    print_section("🎉 DEMO COMPLETE!")
    print("Results saved to:")
    print("  • results/privacy_*/                 (Individual experiments)")
    print("  • results/privacy_comparison.json    (Summary data)")
    print("  • results/visualizations/            (All plots)")
    print("\nKey files to check:")
    print("  📊 results/visualizations/comprehensive_dashboard.png")
    print("  📊 results/visualizations/privacy_tradeoff.png")
    print("  📄 results/privacy_comparison.json")
    
    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Open visualizations to see privacy-accuracy trade-offs")
    print("  2. Read results/privacy_comparison.json for detailed metrics")
    print("  3. Experiment with different privacy levels:")
    print("     python simulation/run_federated.py --privacy high --rounds 15")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
