"""
Unit tests for Privacy-Preserving Search components.
"""
import unittest
import numpy as np
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from client.models.search_model import SearchRankingModel, extract_features
from client.differential_privacy import DifferentialPrivacy
from client.local_train import LocalClient
from server.federated_server import FederatedServer


class TestSearchModel(unittest.TestCase):
    """Test cases for SearchRankingModel."""
    
    def setUp(self):
        """Set up test model."""
        self.model = SearchRankingModel(input_size=3, hidden_size=8)
        self.X = np.random.randn(10, 3)
        self.y = np.random.randint(0, 2, (10, 1))
    
    def test_forward_pass(self):
        """Test forward propagation."""
        output = self.model.forward(self.X)
        
        # Check output shape
        self.assertEqual(output.shape, (10, 1))
        
        # Check output range (sigmoid output should be [0, 1])
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 1))
    
    def test_backward_pass(self):
        """Test gradient computation."""
        self.model.forward(self.X)
        gradients = self.model.backward(self.y)
        
        # Check gradient keys
        expected_keys = {'dW1', 'db1', 'dW2', 'db2'}
        self.assertEqual(set(gradients.keys()), expected_keys)
        
        # Check gradient shapes
        self.assertEqual(gradients['dW1'].shape, self.model.W1.shape)
        self.assertEqual(gradients['db1'].shape, self.model.b1.shape)
        self.assertEqual(gradients['dW2'].shape, self.model.W2.shape)
        self.assertEqual(gradients['db2'].shape, self.model.b2.shape)
    
    def test_train_step(self):
        """Test single training step."""
        loss, gradients = self.model.train_step(self.X, self.y)
        
        # Check loss is a scalar
        self.assertIsInstance(loss, (float, np.floating))
        self.assertGreater(loss, 0)
        
        # Check gradients were computed
        self.assertIsNotNone(gradients)
    
    def test_prediction(self):
        """Test prediction methods."""
        # Predict probabilities
        proba = self.model.predict_proba(self.X)
        self.assertEqual(proba.shape, (10, 1))
        self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1))
        
        # Predict classes
        pred = self.model.predict(self.X)
        self.assertEqual(pred.shape, (10, 1))
        self.assertTrue(np.all(np.isin(pred, [0, 1])))
    
    def test_parameter_management(self):
        """Test getting and setting parameters."""
        # Get parameters
        params = self.model.get_parameters()
        self.assertEqual(set(params.keys()), {'W1', 'b1', 'W2', 'b2'})
        
        # Modify and set back
        params['W1'] = np.ones_like(params['W1'])
        self.model.set_parameters(params)
        
        # Verify change
        self.assertTrue(np.allclose(self.model.W1, np.ones_like(self.model.W1)))


class TestFeatureExtraction(unittest.TestCase):
    """Test cases for feature extraction."""
    
    def test_feature_extraction(self):
        """Test extract_features function."""
        query = "best laptop"
        document = "Top 10 Best Gaming Laptops Under Budget"
        
        features = extract_features(query, document)
        
        # Check shape
        self.assertEqual(features.shape, (3,))
        
        # Check all features are numeric
        self.assertTrue(np.all(np.isfinite(features)))
        
        # Check ranges
        self.assertGreaterEqual(features[0], 0)  # query_len
        self.assertGreaterEqual(features[1], 0)  # doc_len
        self.assertGreaterEqual(features[2], 0)  # overlap
        self.assertLessEqual(features[2], 1)     # overlap should be <= 1


class TestDifferentialPrivacy(unittest.TestCase):
    """Test cases for DifferentialPrivacy."""
    
    def setUp(self):
        """Set up test gradients."""
        self.gradients = {
            'W1': np.random.randn(3, 8) * 0.1,
            'b1': np.random.randn(1, 8) * 0.1,
            'W2': np.random.randn(8, 1) * 0.1,
            'b2': np.random.randn(1, 1) * 0.1
        }
    
    def test_initialization(self):
        """Test DP initialization."""
        dp = DifferentialPrivacy(sigma=0.5, clip_norm=1.0)
        self.assertEqual(dp.sigma, 0.5)
        self.assertEqual(dp.clip_norm, 1.0)
    
    def test_from_privacy_level(self):
        """Test creating DP from predefined levels."""
        levels = ['none', 'low', 'medium', 'high']
        
        for level in levels:
            dp = DifferentialPrivacy.from_privacy_level(level)
            expected_sigma = DifferentialPrivacy.PRIVACY_LEVELS[level]
            self.assertEqual(dp.sigma, expected_sigma)
    
    def test_gradient_clipping(self):
        """Test gradient clipping."""
        dp = DifferentialPrivacy(sigma=0.5, clip_norm=0.1)
        
        # Create large gradient
        large_grads = {
            'W1': np.ones((3, 8)) * 10.0  # Large gradient
        }
        
        clipped = dp.clip_gradients(large_grads)
        
        # Check that gradient was clipped
        grad_norm = np.linalg.norm(clipped['W1'])
        self.assertLessEqual(grad_norm, dp.clip_norm * 1.01)  # Allow small numerical error
    
    def test_noise_addition(self):
        """Test noise addition to gradients."""
        dp_none = DifferentialPrivacy(sigma=0.0)  # No noise
        dp_high = DifferentialPrivacy(sigma=1.0)  # High noise
        
        # Apply no noise
        noisy_none = dp_none.add_noise(self.gradients)
        
        # Apply high noise
        noisy_high = dp_high.add_noise(self.gradients)
        
        # No noise should be identical
        for key in self.gradients.keys():
            self.assertTrue(np.allclose(self.gradients[key], noisy_none[key], atol=1e-10))
        
        # High noise should be different
        for key in self.gradients.keys():
            self.assertFalse(np.allclose(self.gradients[key], noisy_high[key], atol=0.1))
    
    def test_privatize_gradients(self):
        """Test full privatization pipeline."""
        dp = DifferentialPrivacy(sigma=0.5, clip_norm=1.0)
        
        privatized = dp.privatize_gradients(self.gradients)
        
        # Check all keys present
        self.assertEqual(set(privatized.keys()), set(self.gradients.keys()))
        
        # Check shapes preserved
        for key in self.gradients.keys():
            self.assertEqual(privatized[key].shape, self.gradients[key].shape)
    
    def test_privacy_metrics(self):
        """Test privacy metric calculation."""
        dp = DifferentialPrivacy(sigma=0.5)
        
        # Add some noise to populate history
        dp.privatize_gradients(self.gradients)
        
        metrics = dp.get_privacy_metrics()
        
        self.assertEqual(metrics['sigma'], 0.5)
        self.assertGreater(metrics['noise_applications'], 0)


class TestFederatedServer(unittest.TestCase):
    """Test cases for FederatedServer."""
    
    def setUp(self):
        """Set up test server."""
        self.server = FederatedServer()
        
        # Create dummy client gradients
        self.client_gradients = [
            {
                'dW1': np.random.randn(3, 8) * 0.01,
                'db1': np.random.randn(1, 8) * 0.01,
                'dW2': np.random.randn(8, 1) * 0.01,
                'db2': np.random.randn(1, 1) * 0.01
            }
            for _ in range(3)
        ]
    
    def test_get_global_model(self):
        """Test getting global model parameters."""
        params = self.server.get_global_model()
        
        expected_keys = {'W1', 'b1', 'W2', 'b2'}
        self.assertEqual(set(params.keys()), expected_keys)
    
    def test_aggregate_gradients(self):
        """Test gradient aggregation."""
        aggregated = self.server.aggregate_gradients(self.client_gradients)
        
        # Check all keys present
        expected_keys = set(self.client_gradients[0].keys())
        self.assertEqual(set(aggregated.keys()), expected_keys)
        
        # Check shapes
        for key in expected_keys:
            expected_shape = self.client_gradients[0][key].shape
            self.assertEqual(aggregated[key].shape, expected_shape)
    
    def test_weighted_aggregation(self):
        """Test weighted gradient aggregation."""
        weights = [1.0, 2.0, 1.0]  # Second client has 2x weight
        
        aggregated = self.server.aggregate_gradients(
            self.client_gradients,
            weights=weights
        )
        
        # Aggregated should not be identical to simple average
        simple_avg = {
            key: np.mean([g[key] for g in self.client_gradients], axis=0)
            for key in self.client_gradients[0].keys()
        }
        
        # At least one gradient should differ
        differs = False
        for key in aggregated.keys():
            if not np.allclose(aggregated[key], simple_avg[key]):
                differs = True
                break
        
        self.assertTrue(differs, "Weighted aggregation should differ from simple average")
    
    def test_federated_round(self):
        """Test complete federated round."""
        client_metrics = [
            {'accuracy': 0.7, 'loss': 0.5},
            {'accuracy': 0.75, 'loss': 0.45},
            {'accuracy': 0.72, 'loss': 0.48}
        ]
        
        updated_model = self.server.federated_round(
            self.client_gradients,
            client_metrics=client_metrics
        )
        
        # Check round incremented
        self.assertEqual(self.server.current_round, 1)
        
        # Check history updated
        self.assertEqual(len(self.server.history['rounds']), 1)
        self.assertEqual(len(self.server.history['avg_accuracy']), 1)
        
        # Check model returned
        self.assertIsInstance(updated_model, dict)


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*70)
    print("  RUNNING UNIT TESTS")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSearchModel))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestDifferentialPrivacy))
    suite.addTests(loader.loadTestsFromTestCase(TestFederatedServer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
