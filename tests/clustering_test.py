import unittest
from clustering.static_clustering import StaticClustering
from tests.utils import create_block_time_series_dataset

class TestClustering(unittest.TestCase):
    def setUp(self):                
        return super().setUp()

    def tearDown(self):        
        return super().tearDown()

    def test_should_cluster_time_series_dataset_using_parcorr(self):
        ds = create_block_time_series_dataset(5)
        clustering = StaticClustering(ds, "parcorr10")
        clustering.run()
        self.assertGreaterEqual(clustering.silhouette, 0.9)
        
    def test_should_cluster_time_series_dataset_using_gld(self):
        ds = create_block_time_series_dataset(5)
        clustering = StaticClustering(ds, "gld")
        clustering.run()
        self.assertGreaterEqual(clustering.silhouette, 0.6)
      
    def test_should_generate_an_embedding_matrix(self):
        ds = create_block_time_series_dataset(5)
        clustering = StaticClustering(ds, "parcorr4")
        clustering.run()        
        self.assertEqual(
          clustering.embedding_matrix.shape, (*ds.shape[1:], 4))
    
    def test_should_generate_a_clustering_matrix(self):
        ds = create_block_time_series_dataset(5)
        clustering = StaticClustering(ds, "parcorr4")
        clustering.run()        
        self.assertEqual(
          clustering.clustering_matrix.shape, (*ds.shape[1:],))

    def test_should_register_execution_time(self):
        ds = create_block_time_series_dataset(5)
        clustering = StaticClustering(ds, "parcorr10")
        clustering.run()
        self.assertGreaterEqual(clustering.get_total_time(), 0)



if __name__ == '__main__':
    unittest.main()
    