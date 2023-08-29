import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from functools import reduce
import dtw
import core.utils as ut
from core.series_generator import SeriesGenerator
from .noise_generator import NoiseGenerator
from .simple_regressor import LinearRegressor, NonLinearRegressor
from .convolutional_model_invoker import ConvolutionalModelInvoker
from core.tiling.tiling import calculate_centroid
from core.clustering.embedding import get_gld_series_representation_from_dataset
from core.tiling.tile import Tile
from abc import ABC, abstractmethod
from numpy.linalg import LinAlgError

class Learner(ABC):
    model = None
    _is_temporal_model = False
    temporal_length = 10

    def __init__(self, model_directory, model_name,
                 is_temporal_model=False, auto_loading=True):
        self.model_directory = model_directory
        self.model_name = model_name
        self.number_of_training_samples = 10
        self._is_temporal_model = is_temporal_model
        self.output_size = 1
        if auto_loading:
            self.load_model(model_directory, model_name)
        super().__init__()

    def load_model(self, model_directory, model_name):
        self.name = model_name
        self.model_full_path = model_directory + model_name
        # Loads metadata from json file
        json_file = open(model_directory + model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # Load weights into new model
        model.load_weights(model_directory + model_name + '.h5')
        # Loads metadata from database
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self.model = model
        self.reference_dataset = None
        self.centroid = (-1, -1)
        if model.input_shape[1] is not None:
            self.temporal_length = model.input_shape[1]

    def invoke(self, X):
        #from time import time
        #x = time()
        temp = self.get_model().predict(X)
        #print("Tempo de invocação: ", time() - x)
        return temp

    def get_model(self):
        return self.model

    def get_shape(self):
        return self.model.input.shape

    def set_model(self, model):
        self.model = model

    def get_name(self):
        return self.name

    def is_temporal_model(self):
        return self._is_temporal_model

    def get_reference_dataset(self):
        if self.reference_dataset is None:
            dataset_full_path = self.model_directory + self.model_name + '.npy'
            self.reference_dataset = (np.load(dataset_full_path))
            if len(self.reference_dataset.shape) < 3:
                self.reference_dataset = np.reshape(self.reference_dataset,
                                                     (self.reference_dataset.shape[0], 1, 1))
        return self.reference_dataset

    def get_reference_dataset_mock(self):
        dataset_mock = np.full((100, 10, 10), 10.0)
        return dataset_mock

    @abstractmethod
    def evaluate(self, reference_dataset: np.array):
        pass

    @abstractmethod
    def invoke_on_dataset(self, target_dataset):
        pass

    def characterize_dataset_time_series(self, target_dataset):
        print("Calculating GLDs...")
        time, lat, long = target_dataset.shape
        gld_list = get_gld_series_representation_from_dataset(target_dataset)
        gld_list = np.reshape(gld_list, (lat, long, 4))  # Number of GLD parameters = 4
        print("Calculating Centroid...")
        return calculate_centroid(gld_list, (0, 0), (lat, long))

    def compare_series_distances(self, s1, s2, dist_function="dtw"):
        # -. Identify the centroid time series C1 and C2 based on the resulting parameters
        # -. Determine the distance between C1 and C2 (e.g. using euclidian dist between vectors)
        euclidian = lambda x, y: np.linalg.norm(x - y)
        d, cost_matrix, acc_cost_matrix, path = dtw.dtw(s1, s2, dist=euclidian)
        output = d
        #d = output.distance
        if d < 0:
            raise (Exception("Error: Negative dtw Distance"))
        return d

    def compare_dataset_distances(self, original_dataset, compared_dataset, c1, c2):
        # 2. Identify the centroid time series C1 and C2 based on the resulting parameters
        s1 = original_dataset[:, c1[0], c1[1]]
        s2 = compared_dataset[:, c2[0], c2[1]]
        # 3. Determine the distance between C1 and C2 (e.g. using euclidian dist between vectors)
        euclidian = lambda x, y: np.linalg.norm(x - y)
        d, _, _, _ = dtw.dtw(s1, s2, dist=euclidian)
        #d = output.distance
        if d < 0:
            raise(Exception("Error: Negative dtw Distance"))
        return d

    def calculate_centroid_coordinate(self, target_dataset):
        # Characterize each dataset time series (e.g. using GLD or Autoencoder)
        print("Characterizing time series for dataset of shape ", target_dataset.shape)
        return self.characterize_dataset_time_series(target_dataset)

    def get_reference_dataset_centroid_coordinate(self):
        return (0, 0) # todo change for correct procedure
        #if self.centroid == (-1, -1):
        #    self.centroid = self.calculate_centroid_coordinate(self.get_reference_dataset())
        #return self.centroid

    def update_cef(self, noise_level_for_cef, update_models_cef=True, linear_regression=False):
        print("Updating CEF - Model " + self.model_name)
        reference_dataset = self.get_reference_dataset() 
        noise_dataset = reference_dataset.copy()
        parameters_file = self.model_directory + \
                          self.model_name + \
                          "-noise_level-" + str(noise_level_for_cef) \
                                  + ".parameters"

        if update_models_cef or not ut.file_exists(parameters_file):
            # Measures model performance
            distances = []
            error = []
            centroid = self.get_reference_dataset_centroid_coordinate()
            for i in range(noise_level_for_cef):
                print("Evaluating model", self.model_name, " on noise dataset ", i)
                error.append(self.evaluate(noise_dataset))
                distances.append(self.compare_dataset_distances(reference_dataset, noise_dataset,
                                                               centroid, centroid))
                NoiseGenerator().add_noise(noise_dataset)
            with open(parameters_file, "w") as f:
                f.write("distances #" + str(distances) + "\n")
                f.write("error #" + str(error) + "\n")
        else:
            with open(parameters_file) as f:
                distances = f.readline().split("#")[1] #[:-1]
                error = f.readline().split("#")[1] #[:-1]
                distances = eval(distances.strip())
                error = eval(error.strip())

        # Fit line
        if linear_regression:
            r = LinearRegressor(distances, error)
            r.train()
        else:
            while (True):
                try:
                    r = NonLinearRegressor(np.array(distances), np.array(error), label=self.model_name)
                    r.train()
                    break
                except LinAlgError:
                    print("Singular matrix error. Repeating...")
        self.r = r



    def execute_eef(self, dataset, tile: Tile):
        s1 = self.get_reference_learner_series()
        s2 = tile.get_centroid_series()
        x = self.compare_series_distances(s1, s2)
        return self.r.predict(x)

    def get_reference_learner_series(self):
        reference_dataset = self.get_reference_dataset() # Reference model training dataset
        c1 = self.get_reference_dataset_centroid_coordinate()
        return reference_dataset[:, c1[0], c1[1]]


class UnidimensionalLearner(Learner):
    def evaluate(self, target_dataset: np.array):
        time, lat, long = target_dataset.shape
        rmse_vector = []
        for i in range(lat):
            for j in range(long):
                cut_start = 0
                cut_ending = self.temporal_length + self.number_of_training_samples

                from numpy.lib.stride_tricks import sliding_window_view
                #samples = sliding_window_view(X=target_dataset[cut_start:cut_ending, i, j],
                #                              axis=0)
                #X, y = SeriesGenerator().numpy_split_series_x_and_ys_sliding_windows(
                #         target_dataset[cut_start:cut_ending, i, j], self.temporal_length, 1)
                X, y = SeriesGenerator().manual_split_series_into_sliding_windows(
                    target_dataset[cut_start:cut_ending, i, j], self.temporal_length, 1)
                # X = X.reshape((self.number_of_samples, self.temporal_length, 1))
                #output = self.invoke(X)
                output = forecast_lstm(self.model, len(X), X)
                region_rmse = tf.sqrt(tf.math.reduce_mean(tf.losses.mean_squared_error(output, y)))
                rmse_vector.append(region_rmse.numpy())
        average_rmse = reduce(lambda a, b: a + b, rmse_vector) / len(rmse_vector)
        return average_rmse

    def invoke_on_dataset_sequential(self, target_dataset):
        _, lat, long = target_dataset.shape
        time = self.output_size
        output = np.empty((time, lat, long))

        for i in range(lat):
            for j in range(long):
                input = target_dataset[:, i, j]
                input = np.reshape(input, (10, 1))
                out = forecast_lstm(self.model, batch_size=1, X=input)
                #out = self.invoke(input)

                #output[:, i, j] = np.reshape(out, (10))
                output[:, i, j] = out
        return output

    def invoke_on_dataset(self, target_dataset):
        _, lat, long = target_dataset.shape
        time = self.output_size
        output = np.empty((time, lat, long))

        input = np.swapaxes(target_dataset, 0, 1)
        input = np.swapaxes(input, 1, 2)
        input = np.reshape(input, (lat * long, 10, 1))
        out = forecast_lstm(self.model, batch_size=len(input), X=input)
        for i in range(lat):
            for j in range(long):
                output[:, i, j] = out[i * j]
        return output


class MultidimensionalLearner(Learner):
    def evaluate(self, target_dataset: np.array):
        invoker = ConvolutionalModelInvoker()
        return invoker.evaluate_convolutional_model(target_dataset, self)

    def invoke_on_dataset(self, target_dataset):
        invoker = ConvolutionalModelInvoker()
        new_shape = (1,) + target_dataset.shape + (1,)
        extended_target_dataset = np.reshape(target_dataset, new_shape)
        output, _ = invoker.invoke_candidate_model(self, extended_target_dataset)
        return output[0, -1, :, :, 0]

def test_unidimensional_layer():
    directory = "/home/anderson/Dropbox/Doutorado/Tese/Fermatta/DJEnsemble/models/rain/temporal/"
    name = "best_model_C4"
    l = UnidimensionalLearner(directory, name, is_temporal_model=True)
    l.update_cef(1)
    print(l.execute_eef(NoiseGenerator().add_noise(l.get_reference_dataset()), (0, 0)) )

def test_multidimensional_layer():
    directory = "/home/anderson/Programacao/DJEnsemble/Stream-DJEnsemble/models/spatio-temporal/cfsr-all/"
    name = "CFSR-2014.nc-x0=(3, 3)-3x3-('2014-01-01 00:00:00', '2014-03-30 23:45:00')-summer"
    l = MultidimensionalLearner(directory, name, is_temporal_model=False)
    l.update_cef(3)
    print(l.execute_eef(NoiseGenerator().add_noise(l.get_reference_dataset())))

def test_learner(op):
    from core.dataset_manager import DatasetManager
    path = "/home/anderson/Programacao/DJEnsemble/Stream-DJEnsemble/datasets/CFSR-2014.nc"
    ds = DatasetManager(path)
    ds.loadDataset(ds_attribute="TMP_L100")
    data = ds.read_window(t_instant=100, window_size=10)

    if op == "uni":
        directory = "/home/anderson/Programacao/DJEnsemble/Stream-DJEnsemble/models/temporal/cfsr-all/"
        name = "CFSR-2014.nc-x0=(53, 1)-1x1-summer"
        l = UnidimensionalLearner(directory, name, is_temporal_model=True)
        lt, lg = 53, 1
        size = 1
        data = np.load(directory + "CFSR-2014.nc-x0=(53, 1)-1x1-summer.npy")
        data = np.reshape(data, newshape=(data.shape[0], 1, 1))
        y = data[10, lt:lt + size, lg:lg + size]
        input = data[:10, lt:lt + size, lg:lg + size]
    else:
        directory = "/home/anderson/Programacao/DJEnsemble/Stream-DJEnsemble/models/spatio-temporal/cfsr/"
        lt, lg = 55, 2
        size = 3
        name = "CFSR-2014.nc-x0=(55, 2)-3x3-('2014-01-01 00:00:00', '2014-03-30 23:45:00')-summer"
        l = MultidimensionalLearner(directory, name, is_temporal_model=False)
        data = np.load(directory + name + ".npy")
        y = data[10]
        input = data[:10]

    X = input
    yhat = forecast_conv_lstm(l.model, 1, X)
    print(y - yhat)

def forecast_lstm(model, batch_size, X):
    X = X.reshape(batch_size, 10, 1)
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[:, 0]

def forecast_conv_lstm(model, batch_size, X):
    X = X.reshape(*X.shape[:], 1)
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[..., 0]