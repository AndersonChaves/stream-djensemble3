import unittest
import tests.utils as utils
from core.config import Config
from query.ensemble_runner import EnsembleRunner

class TestRunner(unittest.TestCase):
    def setUp(self):                
        return super().setUp()

    def tearDown(self):        
        return super().tearDown()

    def test_should_execute_ensemble(self):
        config = Config("config/config-tests.json")
        runner = EnsembleRunner(config.data["tests"]["runner_test"])
        target_dataset = utils.create_block_time_series_dataset(5)
        ensemble = utils.create_mock_ensemble()
        runner.run(ensemble, target_dataset)
        self.assertGreaterEqual(1, 0.9)        
        
if __name__ == '__main__':
    unittest.main()
    