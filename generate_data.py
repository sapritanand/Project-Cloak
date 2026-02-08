"""
Generate synthetic search data for federated learning simulation.
Each device gets different data to simulate real-world heterogeneity.
"""
import json
import random
import numpy as np
from pathlib import Path


class SearchDataGenerator:
    """Generate realistic synthetic search click data."""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Sample queries and documents
        self.queries = [
            "best laptop", "gaming laptop", "budget laptop", "laptop for work",
            "python tutorial", "learn python", "python course", "python basics",
            "healthy recipes", "quick dinner", "breakfast ideas", "meal prep",
            "running shoes", "nike shoes", "athletic shoes", "workout shoes",
            "travel destinations", "cheap flights", "vacation spots", "weekend getaway"
        ]
        
        self.documents = [
            "Top 10 Gaming Laptops Under $1000",
            "Best Budget Laptops for Students 2024",
            "Professional Workstation Laptops Review",
            "Gaming Laptop Buying Guide",
            "Python Programming Complete Course",
            "Learn Python in 30 Days - Free Tutorial",
            "Python Basics for Beginners",
            "Advanced Python Programming Guide",
            "10 Healthy Dinner Recipes Under 30 Minutes",
            "Quick Breakfast Ideas for Busy Mornings",
            "Meal Prep Guide for Beginners",
            "Healthy Recipes for Weight Loss",
            "Best Running Shoes 2024 Review",
            "Nike vs Adidas: Running Shoe Comparison",
            "Budget Athletic Shoes Guide",
            "Top Workout Shoes for Gym",
            "Top 15 Travel Destinations 2024",
            "How to Find Cheap Flights",
            "Best Weekend Getaway Spots",
            "Budget Vacation Ideas"
        ]
    
    def _compute_relevance(self, query, document):
        """
        Compute relevance score based on keyword overlap.
        This creates realistic click patterns.
        """
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        overlap = len(query_words.intersection(doc_words))
        max_overlap = len(query_words)
        
        if max_overlap == 0:
            return 0.1
        
        base_relevance = overlap / max_overlap
        # Add some randomness
        return min(1.0, base_relevance + random.uniform(-0.1, 0.1))
    
    def generate_device_data(self, num_samples=100, device_id=0):
        """
        Generate data for a single device.
        Each device has biased data (simulates user preferences).
        """
        # Each device has preference for certain query types
        device_bias = device_id % len(self.queries)
        
        samples = []
        for _ in range(num_samples):
            # Biased sampling towards certain queries
            if random.random() < 0.6:
                query_idx = (device_bias + random.randint(0, 2)) % len(self.queries)
            else:
                query_idx = random.randint(0, len(self.queries) - 1)
            
            query = self.queries[query_idx]
            doc_idx = random.randint(0, len(self.documents) - 1)
            document = self.documents[doc_idx]
            
            # Compute relevance and generate click
            relevance = self._compute_relevance(query, document)
            # Click probability based on relevance
            click = 1 if random.random() < relevance else 0
            
            samples.append({
                "query": query,
                "document": document,
                "click": click,
                "relevance": round(relevance, 3)
            })
        
        return samples
    
    def save_device_data(self, output_dir, num_devices=5, samples_per_device=100):
        """Generate and save data for multiple devices."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        device_stats = []
        
        for device_id in range(num_devices):
            samples = self.generate_device_data(samples_per_device, device_id)
            
            # Calculate statistics
            click_rate = sum(s['click'] for s in samples) / len(samples)
            avg_relevance = sum(s['relevance'] for s in samples) / len(samples)
            
            device_stats.append({
                "device_id": device_id,
                "num_samples": len(samples),
                "click_rate": round(click_rate, 3),
                "avg_relevance": round(avg_relevance, 3)
            })
            
            # Save to file
            device_file = output_path / f"device_{device_id}_data.json"
            with open(device_file, 'w') as f:
                json.dump(samples, f, indent=2)
            
            print(f"✓ Device {device_id}: {len(samples)} samples, "
                  f"click_rate={click_rate:.2%}, avg_relevance={avg_relevance:.3f}")
        
        # Save summary statistics
        summary_file = output_path / "data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(device_stats, f, indent=2)
        
        print(f"\n✓ Data generated for {num_devices} devices")
        print(f"✓ Files saved to: {output_path}")
        
        return device_stats


if __name__ == "__main__":
    generator = SearchDataGenerator(seed=42)
    stats = generator.save_device_data(
        output_dir="client/local_data",
        num_devices=5,
        samples_per_device=100
    )
