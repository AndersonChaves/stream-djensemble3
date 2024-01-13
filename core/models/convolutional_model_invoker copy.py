from functools import reduce
import core.model_training as mt
import core.dataset_manager as dataset_manager
import tensorflow as tf
import numpy as np
from .series_generator import SeriesGenerator

class ConvolutionalModelInvoker:
    def __init__(self):
        pass

    def evaluate_convolutional_model(self, target_dataset, model):
        X, y = SeriesGenerator().numpy_split_series_x_and_ys_sliding_windows(
            target_dataset, model.temporal_length, 1)

        predicted_frame_series = mt.forecast_conv_lstm(model.model, len(X), X)
        y = np.reshape(y, (len(y), *y.shape[-2:]))
        region_rmse = tf.sqrt(
            tf.math.reduce_mean(
                tf.losses.mean_squared_error(predicted_frame_series, y)
            )
        )
        average_rmse = region_rmse

        return average_rmse

    def evaluate_convolutional_model_sequential(self, target_dataset, model):
        frame_series = SeriesGenerator().generate_frame_series(
            target_dataset, model.temporal_length)
        # FOR EVERY FRAME SERIES IN TILE
        rmse_by_frame = []
        for s, (frame_series_input, frame_series_output) in enumerate(frame_series):
            # gets a prediction from a model ensemble, the average pred of different models
            predicted_frame_series = self.averaging_ensemble(frame_series_input, [model])
            if s % 10 == 0:
                print("Evaluated frame ", s)
            # computes the rmse for the frame.
            last_output_frame = frame_series_output[:, -1, :, :]
            last_predicted_frame = predicted_frame_series[:, -1, :, :, :]
            last_output_frame = last_output_frame.reshape(last_predicted_frame.shape)
            loss = tf.sqrt(tf.math.reduce_mean(tf.losses.mean_squared_error(last_output_frame, last_predicted_frame)))
            rmse_by_frame.append(loss.numpy())
        average_rmse = reduce(lambda a, b: a + b, rmse_by_frame) / len(rmse_by_frame)
        return average_rmse

    def averaging_ensemble(self, frame_series, learner_list, weights=None):
        if weights is None:
            weights = [1 for i in range(len(learner_list))]

        y_m = []
        length_x = []
        for i, learner in enumerate(learner_list):
            result_y, length_x = self.invoke_candidate_model(learner, frame_series)
            y_m.append(result_y)

        sum = tf.zeros_like(length_x)

        # The average prediction of different models
        # Isolate this part to adapt to model stacking
        for i, y in enumerate(y_m):
            sum = sum + y * weights[i]
        total = reduce(lambda a, b: a + b, weights) # sum all weights
        return sum / total

    def invoke_candidate_model(self, learner, query):
        shape = learner.get_shape()
        # number of iterations in x and y axis
        x_size, y_size = int(shape[2]), int(shape[3])  # Size of the models input frame

        # Duplicates the dataset when necessary to fit the models input
        temp_query = dataset_manager.extend_dataset(x_size, y_size, query)

        # How many times does the model fit on that dimension: query_dim / model_dim
        length_x = int(temp_query.shape[2] / int(shape[2]))
        length_y = int(temp_query.shape[3] / int(shape[3]))

        yp = np.zeros((temp_query.shape))
        x = query.shape
        list = []
        for x in range(length_x):
            for y in range(length_y):
                # Get the corresponding query data and predict
                lat_i, lat_e = x * int(shape[2]), (x + 1) * int(shape[2])
                lon_i, lon_e = y * int(shape[3]), (y + 1) * int(shape[3])
                prediction = learner.invoke(temp_query[:, :, lat_i:lat_e, lon_i:lon_e, :])
                list.append((lat_i, lat_e, lon_i, lon_e, prediction))

        for e in list:
            # Get the corresponding query data and predict
            yp[:, :, e[0]:e[1], e[2]:e[3], :] = e[4]
        output_frame_series = yp[:, :, :query.shape[2], :query.shape[3], :]
        #output_frame        = output_frame_series[:,9, :, :, :]
        # Cuts the predicted section corresponding to the original query only
        return output_frame_series, x