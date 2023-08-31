import numpy as np
import sys
print(sys.version)
from numpy.lib.stride_tricks import sliding_window_view

class SeriesGenerator:
    # Generate series of specified length for convolutional models
    def generate_frame_series(self, data, temporal_length):
        for i in range(data.shape[0] - temporal_length):
            X = np.zeros((1, temporal_length, data.shape[1], data.shape[2], 1), dtype=float)
            Y = np.zeros((1, temporal_length, data.shape[1], data.shape[2], 1), dtype=float)
            X[0, :, :, :, 0] = data[i:i + temporal_length, :, :]
            Y[0, :, :, :, 0] = data[i + 1:i + temporal_length + 1, :, :]
            yield X, Y

    # Former split_sequence function
    def manual_split_series_into_sliding_windows(self, sequence, n_steps_in, n_steps_out):
        # X is the series inputs, y is the expected outputs for each series
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(sequence):
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def numpy_split_series_x_and_ys_sliding_windows(self, sequence, n_steps_in, n_steps_out):
        sequence_shape = list(sequence.shape)
        ws = tuple([n_steps_in] + sequence_shape[1:])
        X = sliding_window_view(sequence[:-n_steps_out], window_shape=ws)

        ws = tuple([n_steps_out] + sequence_shape[1:])
        y = sliding_window_view(sequence[n_steps_in:], window_shape=ws)

        return np.array(X)[:, 0, 0, :], np.array(y)[:, 0, 0, :]

    def numpy_split_series_into_sliding_windows(self, sequence, window_size):
        sequence_shape = list(sequence.shape)
        ws = tuple([window_size] + sequence_shape[1:])
        X = sliding_window_view(sequence, window_shape=ws)
        return np.array(X)[:, 0, 0, :]

    def numpy_split_series_into_tumbling_windows(self, sequence, window_size): # todo - finish
        sequence_shape = list(sequence.shape)
        ws = tuple([window_size] + sequence_shape[1:])
        X = sliding_window_view(sequence, window_shape=ws)
        return np.array(X)[:, 0, 0, :]


    def split_series_into_tumbling_windows(self, sequence, n_steps_in, n_steps_out):
        # X is the series inputs, y is the expected outputs for each series
        X, y = list(), list()
        for i in range(0, len(sequence), n_steps_in):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(sequence):
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
