import numpy as np
from pandas import DataFrame
from pandas import Series
from pandas import concat
from tensorflow.keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from math import sqrt
from matplotlib import pyplot
import numpy
from numpy.lib.stride_tricks import sliding_window_view
import core.view as view

from core.learner import MultidimensionalLearner
import core.model_training as mt

class ConvLstmLearner(MultidimensionalLearner):
    def __init__(self, model_directory, model_name,
                  auto_loading = False):
        is_temporal_model = True
        self.series_size = -1
        super().__init__(model_directory, model_name,
                          is_temporal_model=is_temporal_model, auto_loading=auto_loading)
        self.update_training_options()
        self.update_architecture()

    def update_training_options(self, differentiate=False,
              train_as_stateful=False, scale_data=False):
        self.differentiate     = differentiate
        self.train_as_stateful = train_as_stateful
        self.scale_data        = scale_data
        self.scaler            = None

    def update_architecture(self, neurons=32, nb_epochs=100,
                            batch_size=1, number_of_hidden_layers=1):
        self.neurons   = neurons
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.number_of_hidden_layers = number_of_hidden_layers

    def train(self, training_series, series_size):
        self.series_size = series_size

        x_train, x_val, y_train, y_val, train_dataset, val_dataset = mt.conv_transform_supervised(training_series, series_size)
        supervised_values = [x_train, y_train]

        if self.scale_data:
            # transform the scale of the data
            train_scaled, scaler = mt.scale(supervised_values)
        else:
            train_scaled, scaler = supervised_values, None

        # fit the model
        conv_lstm_model = mt.fit_conv_lstm(train_scaled, batch_size=self.batch_size,
                                           nb_epoch=self.nb_epochs,
                              neurons=self.neurons, is_stateful=self.train_as_stateful,
                                 number_of_hidden_layers=self.number_of_hidden_layers)

        if self.train_as_stateful:
            train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
            conv_lstm_model.predict(train_reshaped, batch_size=1)
        self.model = conv_lstm_model

    def predict(self, conv_lstm_model, raw_testing_series, differentiate=False,
                scaler=None, series_size = 10):
        X_test, y_test = raw_testing_series[0], raw_testing_series[1]
        frame_shp = X_test.shape[1:3]
        # Define window size
        ws = list(frame_shp)
        ws = tuple([series_size+1] + ws + [1])
        X = sliding_window_view(X_test, window_shape=ws)
        X = X[:, 0, 0, 0, :, :, :]

        X, y = X[:, 0:-1], X[:, -1]
        yhat = conv_lstm_model.predict(X)

        print('Predicted: ', yhat[:], '\nExpected: ', y)

        # report performance
        rmse = sqrt(mean_squared_error(yhat.flatten(), y.flatten()))
        print('Test RMSE: %.3f' % rmse)

        # line plot of observed vs predicted
        fig, ax = pyplot.subplots()
        average_y = np.sqrt(np.power(y, 2))
        average_y = np.mean(np.mean(average_y[:, :, :, 0], axis=1), axis=1)
        pyplot.plot(average_y)

        average_yhat = np.sqrt(np.power(yhat, 2))
        average_yhat = np.mean(np.mean(average_yhat[:, :, :, 0], axis=1), axis=1)
        pyplot.plot(average_yhat)
        pyplot.show()
        fig.savefig(self.model_directory + self.model_name + '.png', dpi=120)
        file = self.model_directory + self.model_name + '.txt'
        with open(file, "w") as f:
            f.write('Test RMSE: %.3f' % rmse)