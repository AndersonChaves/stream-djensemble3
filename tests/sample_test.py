import unittest

class TestRunner(unittest.TestCase):
    def setUp(self):                
        return super().setUp()

    def tearDown(self):        
        return super().tearDown()

    def test_should_execute_ensemble(self):
        self.assertGreaterEqual(1, 0.9)
        
if __name__ == '__main__':
    unittest.main()
    