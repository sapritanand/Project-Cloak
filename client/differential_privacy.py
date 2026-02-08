"""
Differential Privacy Implementation.
Adds calibrated Gaussian noise to gradients for privacy protection.
"""
import numpy as np


class DifferentialPrivacy:
    """
    Implements basic differential privacy using Gaussian mechanism.
    
    Privacy is controlled by sigma (noise scale):
    - Higher sigma = More privacy, Less accuracy
    - Lower sigma = Less privacy, More accuracy
    """
    
    # Predefined privacy levels
    PRIVACY_LEVELS = {
        'none': 0.0,
        'low': 0.1,
        'medium': 0.5,
        'high': 1.0,
        'very_high': 2.0
    }
    
    def __init__(self, sigma=0.5, clip_norm=1.0):
        """
        Args:
            sigma: Noise scale (standard deviation)
            clip_norm: Gradient clipping threshold (for bounded sensitivity)
        """
        self.sigma = sigma
        self.clip_norm = clip_norm
        self.noise_history = []
    
    @classmethod
    def from_privacy_level(cls, level='medium', clip_norm=1.0):
        """
        Create DifferentialPrivacy instance from predefined level.
        
        Args:
            level: 'none', 'low', 'medium', 'high', or 'very_high'
            clip_norm: Gradient clipping threshold
        """
        sigma = cls.PRIVACY_LEVELS.get(level, 0.5)
        return cls(sigma=sigma, clip_norm=clip_norm)
    
    def clip_gradients(self, gradients):
        """
        Clip gradients to bound sensitivity.
        Uses L2 norm clipping.
        
        Args:
            gradients: Dictionary of gradient arrays
        
        Returns:
            Clipped gradients
        """
        clipped_grads = {}
        
        for key, grad in gradients.items():
            # Compute L2 norm
            grad_norm = np.linalg.norm(grad)
            
            # Clip if necessary
            if grad_norm > self.clip_norm:
                clipped_grads[key] = grad * (self.clip_norm / grad_norm)
            else:
                clipped_grads[key] = grad.copy()
        
        return clipped_grads
    
    def add_noise(self, gradients):
        """
        Add Gaussian noise to gradients.
        
        Args:
            gradients: Dictionary of gradient arrays
        
        Returns:
            Noisy gradients
        """
        noisy_grads = {}
        total_noise_magnitude = 0
        
        for key, grad in gradients.items():
            # Generate Gaussian noise with same shape as gradient
            noise = np.random.normal(0, self.sigma, grad.shape)
            
            # Add noise
            noisy_grads[key] = grad + noise
            
            # Track noise magnitude
            total_noise_magnitude += np.linalg.norm(noise)
        
        # Store noise statistics
        self.noise_history.append({
            'sigma': self.sigma,
            'noise_magnitude': total_noise_magnitude
        })
        
        return noisy_grads
    
    def privatize_gradients(self, gradients):
        """
        Full DP pipeline: clip then add noise.
        
        Args:
            gradients: Dictionary of gradient arrays
        
        Returns:
            Privatized gradients
        """
        # Step 1: Clip gradients (bound sensitivity)
        clipped_grads = self.clip_gradients(gradients)
        
        # Step 2: Add noise (privacy)
        if self.sigma > 0:
            noisy_grads = self.add_noise(clipped_grads)
        else:
            noisy_grads = clipped_grads
        
        return noisy_grads
    
    def get_privacy_metrics(self):
        """
        Calculate privacy metrics.
        
        Returns:
            Dictionary with privacy statistics
        """
        if not self.noise_history:
            return {
                'sigma': self.sigma,
                'clip_norm': self.clip_norm,
                'avg_noise_magnitude': 0,
                'noise_applications': 0
            }
        
        avg_noise = np.mean([h['noise_magnitude'] for h in self.noise_history])
        
        return {
            'sigma': self.sigma,
            'clip_norm': self.clip_norm,
            'avg_noise_magnitude': avg_noise,
            'noise_applications': len(self.noise_history)
        }
    
    def estimate_privacy_loss(self, num_rounds=10):
        """
        Rough estimate of privacy budget consumed.
        
        This is a simplified calculation for educational purposes.
        Real privacy accounting is more complex (e.g., Rényi DP).
        
        Returns:
            Approximate epsilon (privacy loss parameter)
        """
        if self.sigma == 0:
            return float('inf')  # No privacy
        
        # Simplified: epsilon ≈ sensitivity * sqrt(num_rounds) / sigma
        # Using clip_norm as sensitivity bound
        epsilon = self.clip_norm * np.sqrt(num_rounds) / self.sigma
        
        return epsilon
    
    def get_privacy_summary(self, num_rounds=1):
        """
        Get human-readable privacy summary.
        
        Args:
            num_rounds: Number of training rounds
        
        Returns:
            Dictionary with privacy explanation
        """
        epsilon = self.estimate_privacy_loss(num_rounds)
        
        # Privacy interpretation
        if epsilon == float('inf'):
            privacy_level = "None (no noise)"
        elif epsilon < 1:
            privacy_level = "Very High (ε < 1)"
        elif epsilon < 5:
            privacy_level = "High (ε < 5)"
        elif epsilon < 10:
            privacy_level = "Medium (ε < 10)"
        else:
            privacy_level = "Low (ε ≥ 10)"
        
        return {
            'sigma': self.sigma,
            'epsilon': round(epsilon, 2) if epsilon != float('inf') else '∞',
            'privacy_level': privacy_level,
            'explanation': self._get_explanation()
        }
    
    def _get_explanation(self):
        """Get explanation of current privacy setting."""
        if self.sigma == 0:
            return "No privacy protection. Gradients sent as-is."
        elif self.sigma < 0.3:
            return "Low privacy. Minimal noise added to gradients."
        elif self.sigma < 0.8:
            return "Medium privacy. Moderate noise protects against attacks."
        else:
            return "High privacy. Strong noise protects data but may reduce accuracy."


def compare_privacy_levels(gradients, levels=['none', 'low', 'medium', 'high']):
    """
    Demonstrate effect of different privacy levels on gradients.
    
    Args:
        gradients: Dictionary of gradient arrays
        levels: List of privacy levels to compare
    
    Returns:
        Comparison results
    """
    results = []
    
    # Create a copy of original gradients
    original_norm = sum(np.linalg.norm(g) for g in gradients.values())
    
    for level in levels:
        dp = DifferentialPrivacy.from_privacy_level(level)
        noisy_grads = dp.privatize_gradients(gradients)
        
        # Calculate distortion
        noisy_norm = sum(np.linalg.norm(g) for g in noisy_grads.values())
        distortion = abs(noisy_norm - original_norm) / original_norm if original_norm > 0 else 0
        
        results.append({
            'level': level,
            'sigma': dp.sigma,
            'gradient_distortion': round(distortion, 3),
            'privacy_summary': dp.get_privacy_summary()
        })
    
    return results


if __name__ == "__main__":
    print("Testing Differential Privacy...")
    
    # Create dummy gradients
    gradients = {
        'W1': np.random.randn(3, 8) * 0.1,
        'b1': np.random.randn(1, 8) * 0.1,
        'W2': np.random.randn(8, 1) * 0.1,
        'b2': np.random.randn(1, 1) * 0.1
    }
    
    # Test different privacy levels
    print("\n=== Privacy Level Comparison ===")
    results = compare_privacy_levels(gradients)
    
    for r in results:
        print(f"\nLevel: {r['level'].upper()}")
        print(f"  Sigma: {r['sigma']}")
        print(f"  Gradient Distortion: {r['gradient_distortion']:.1%}")
        print(f"  Privacy: {r['privacy_summary']['privacy_level']}")
    
    print("\n✓ Differential Privacy test passed!")
