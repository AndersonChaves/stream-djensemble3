from sklearn.cluster import Birch
import core.categorization as ct
import numpy as np
import core.utils as ut
from .config_manager import ConfigManager
from .query_manager import QueryManager
from .series_generator import SeriesGenerator
import time

class ClusterManager():
    def __init__(self, config_manager: ConfigManager,
                   n_clusters=4, notifier_list = None, benchmarks=None):
        self.notifier_list = []
        if notifier_list is not None:
            self.notifier_list = notifier_list
        self.benchmarks = benchmarks
        self.config_manager = config_manager
        self.clustering_algorithm = config_manager.get_config_value("clustering_algorithm")
        self.bool_load_global_clustering_from_file = config_manager.get_config_value(
            "load_global_clustering_from_file") == 's'
        self.embedding_method = config_manager.get_config_value("global_embedding_method")
        self.clustering_behavior = self.config_manager.get_config_value("series_clustering_behavior")
        self.tiling_metadata = {}
        if self.clustering_algorithm=="birch":
            #self.birch = Birch(branching_factor=6, n_clusters=2, threshold=15)
            self.birch = Birch(n_clusters=n_clusters)
        self.tiling_is_updated = False
        self.clustering_directory = "results/clustering/"

    def is_global_clustering(self) -> bool:
        return self.clustering_behavior == 'global'

    def update_clustering_local(self, update_dataset: np.array,
                                query_manager: QueryManager):
        clustering_benchmarks = {}
        for query_id in query_manager.get_all_query_ids():
            start_update_clustering = time.time()
            continuous_query = query_manager.get_continuous_query(query_id)
            x1, x2 = continuous_query.get_query_endpoints()
            query_dataset = update_dataset[:, x1[0]:x2[0], x1[1]:x2[1]]
            query_clulstering_benchmarks = continuous_query.update_clustering(query_dataset)
            for key, value in query_clulstering_benchmarks.items():
                clustering_benchmarks[query_id + "_" + key] = value

            clustering_time =time.time() - start_update_clustering
            self.log("--CLUSTERING--Updated local clustering: query" + str(query_id) + ":" + \
              str(clustering_time))

            self.log("--TILING--Number of tiles: query" + str(query_id) + ":" + \
                     str(continuous_query.get_current_number_of_tiles()))
        return clustering_benchmarks


    def update_global_clustering(self, update_dataset: np.array,
                                 embeddings_list=None):
        clustering_benchmarks = {}
        start_update_clustering = time.time()
        if embeddings_list is None:
            embeddings_list = ct.get_embedded_series_representation(update_dataset, method=self.embedding_method)
            ct.normalize_embedding_list(embeddings_list)
        else:
            shp = embeddings_list.shape
            embeddings_list = np.reshape(embeddings_list, (shp[0] * shp[1], shp[2]))

        self.global_series_embedding = embeddings_list
        new_emb_shape = update_dataset.shape[1:] +  tuple([embeddings_list.shape[-1]])
        self.clustering = self.cluster_embeddings_using_birch(self.global_series_embedding)
        self.clustering = np.reshape(self.clustering, newshape=new_emb_shape[:-1])
        self.global_series_embedding = np.reshape(embeddings_list, newshape=new_emb_shape)

        clustering_time = time.time() - start_update_clustering
        self.log("--GLOBAL CLUSTERING--Updated global clustering, method " + self.embedding_method + \
                 ': ' + str(clustering_time))

        start_update_tiling = time.time()
        self.perform_global_static_tiling(update_dataset)
        tiling_time = time.time() - start_update_tiling

        clustering_benchmarks["GLOBAL_CLUSTERING_METHOD"] = self.embedding_method
        clustering_benchmarks["GLOBAL_CLUSTERING_TIME"] = clustering_time
        clustering_benchmarks["GLOBAL_N_CLUSTERS"] = len(np.unique(self.clustering))

        clustering_benchmarks["GLOBAL_TILING_TIME"] = tiling_time
        clustering_benchmarks["GLOBAL_TILING_METHOD"] = self.config_manager.get_config_value("global_tiling_method")
        clustering_benchmarks["GLOBAL_N_TILES"] = len(self.get_tiling_metadata().keys())
        self.tiling_is_updated = True
        return clustering_benchmarks


    def cluster_embeddings_using_birch(self, emb_list: np.array):
        self.birch = self.birch.partial_fit(emb_list)
        clustering = self.birch.predict(emb_list)
        return clustering

    def try_loading_clustering_from_file(self, label) -> bool:
        file_name = ut.get_file_name_from_path(self.config_manager.get_config_value("dataset_path"))
        file_name += "-" + self.embedding_method + "-" + label
        clustering_file_name = self.clustering_directory + file_name + ".clustering.npy"
        embedding_file_name = self.clustering_directory + file_name + ".embedding.npy"
        silhouette_file_name = self.clustering_directory + file_name + ".silhouette"
        self.log("Try Cluster Loading")
        if self.bool_load_global_clustering_from_file and ut.file_exists(clustering_file_name):
            self.log("Loading Clustering: " + clustering_file_name)
            self.global_series_embedding = np.load(embedding_file_name)
            self.clustering              = np.load(clustering_file_name)
            with open(silhouette_file_name, "r") as f:
                ln =f.readline()
                self.best_silhouette = float(ln)
            return True
        else:
            return False

    def save_clustering(self, label):

        file_name = ut.get_file_name_from_path(self.config_manager.get_config_value("dataset_path"))
        file_name += "-" + self.embedding_method + "-" + label

        clustering_file_name = self.clustering_directory + file_name + ".clustering"
        embedding_file_name = self.clustering_directory + file_name + ".embedding"
        silhouette_file_name = self.clustering_directory + file_name + ".silhouette"

        self.log("Saving Clustering: " + clustering_file_name)

        np.save(clustering_file_name, self.clustering)
        np.save(embedding_file_name, self.global_series_embedding)
        with open(silhouette_file_name, "w") as f:
            f.write(str(self.best_silhouette))

    def perform_global_static_clustering(self, data_frame_series: np.array, label):
        if not self.try_loading_clustering_from_file(label):
            self.log("Start static clusterization")
            start_initialize_clustering = time.time()
            self.global_series_embedding, self.clustering, self.best_silhouette = ct.cluster_dataset(data_frame_series, self.embedding_method)
            self.clustering = np.reshape(self.clustering, newshape=data_frame_series.shape[1:])
            new_emb_shape = data_frame_series.shape[1:] + tuple([self.global_series_embedding.shape[1]]) # todo verify if it works for parcorr
            self.global_series_embedding = np.reshape(self.global_series_embedding, newshape=new_emb_shape)
            self.log("--GLOBAL CLUSTERING--Performed static global clustering: " + \
                     str(time.time() - start_initialize_clustering))
            self.save_clustering(label)
            self.tiling_is_updated = False

    def perform_global_static_tiling(self, data_frame_series: np.array):
        # self.perform_global_static_clustering(data_frame_series)
        start_initialize_tiling = time.time()
        if self.global_series_embedding is None:
            raise("Error: Must perform static clusterization before static tiling")
        self.log("Start static clusterization")
        emb_frame = self.global_series_embedding
        tiling_method = self.config_manager.get_config_value("global_tiling_method")
        purity = float(self.config_manager.get_config_value("global_min_tiling_purity_rate"))
        self.log("--GLOBAL STATIC TILING--Performing tiling. " + ": ")
        self.tiling, self.tiling_metadata = ct.perform_tiling(data_frame_series, emb_frame,
                                                              self.clustering, tiling_method, purity)
        self.log("Static tiling time: " + str(time.time() - start_initialize_tiling))
        self.tiling_is_updated = True
        return self.tiling, self.tiling_metadata

    def get_tiling_metadata(self):
        # if not self.tiling_is_updated:
        #     emb_frame = self.global_series_embedding
        #     tiling_method = self.config_manager.get_config_value("global_tiling_method")
        #     purity = float(self.config_manager.get_config_value("global_min_tiling_purity_rate"))
        #     self.tiling, self.tiling_metadata = ct.perform_tiling(target_dataset=emb_frame, self.clustering, tiling_method, purity)
        #     self.tiling_is_updated = True
        if not self.tiling_is_updated:
            raise(Exception("Tiling may not be updated."))
        return self.tiling_metadata

    def get_current_number_of_tiles(self):
        return len(self.get_tiling_metadata().keys())

    def clusters_from_series(self, data_series: np.array):
        gld_list = ct.get_gld_series_representation_from_dataset(data_series)
        ct.normalize_embedding_list(gld_list)
        clustering = self.birch.predict(data_series)
        return clustering

    def log(self, msg):
        for notifier in self.notifier_list:
            notifier.notify(msg)

def test_prepare_dummy_array():
    # Prepare dummy array
    s1 = np.array([i + 1 for i in range(10)] * 10)
    s2 = np.array([i + 1 if i % 2 == 0 else -i for i in range(10)] * 10)
    dummy_array = np.array((
        (s1, s1, s1, s1, s1),
        (s1, s1, s1, s1, s1),
        (s2, s2, s2, s2, s2),
        (s2, s2, s2, s2, s2),
        (s2, s2, s2, s2, s2)
    ))
    dummy_array = np.moveaxis(dummy_array, 2, 0)
    return dummy_array

def test_perform_global_clustering(dummy_array):
    gld_list, global_clustering = ClusterManager(
        ConfigManager("../experiment-metadata/djensemble-exp1.config")).perform_global_static_clustering(dummy_array[:10])
    return global_clustering

def test_prepare_dataset_for_online_clustering(dummy_array):
    # -- Perform online continuous Clustering
    # Turing dataset into windows
    frame_window_series, _ = SeriesGenerator().split_series_into_tumbling_windows(
        dummy_array, 10, n_steps_out=0)

    return frame_window_series

def test_dummy_array():
    dummy_array = test_prepare_dummy_array()
    # global_clustering = test_perform_global_clustering(dummy_array)
    frame_window_series = test_prepare_dataset_for_online_clustering(dummy_array)
    local_cls_manager = ClusterManager(ConfigManager("../experiment-metadata/djensemble-exp1.config"))

    local_clustering = []
    history_gld_list = np.empty((0, 4))
    for i, fs in enumerate(frame_window_series):
        gld_list, local_clustering = local_cls_manager.update_global_clustering(fs)

        history_gld_list = np.concatenate((history_gld_list, gld_list), axis=0)
        history_clustering = local_cls_manager.birch.predict(history_gld_list)

        x = history_gld_list[:, 0]
        y = history_gld_list[:, 2]
        view.save_figure_as_scatter_plot(x, y, history_clustering, "clustering-" + str(i))
        view.save_figure_from_matrix(np.reshape(local_clustering, (5, 5)), "matrix-" + str(i))

        # # Annotate gld series positions
        # for j, point in enumerate(gld_list):
        #     plt.annotate(str(j), (point[0], point[1]), fontsize=5)
        # # Print leaves
        #
        # leaves = local_cls_manager.birch.subcluster_centers_
        # plt.ylim(0, 100)
        # plt.xlim(0, 100)
        # plt.scatter(leaves[:, 0], leaves[:, 1], facecolors='none', edgecolors='black')
        #
        # plt.savefig(str(i) + ".png")
        # plt.clf()

    # print((global_clustering - local_clustering).sum()) # noqa
    print("local cl: ", local_clustering)
    print("global cl: ", global_clustering)
    coincidence_1 = [i for i, x in enumerate(local_clustering[:10]) if x in local_clustering[10:]]
    coincidence_2 = [i for i, x in enumerate(local_clustering[10:]) if x in local_clustering[:10]]
    print("1: ", coincidence_1)
    print("2: ", coincidence_2)

    import clustering.clustering as clustering
    # categorization.print_array(gld_list)

    # print("Intersection: ", inter, "Intersection length", len(inter))
    # print("Group: ", )
    # incorrect = [g for g in local_clustering[10:] if g in group_1]
    # print("Group Repetition: ", incorrect, "Length: ", len(incorrect))

if __name__ == '__main__':
    test_dummy_array()

