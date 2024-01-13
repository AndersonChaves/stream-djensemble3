from pandas import DataFrame
from pandas import Series
from pandas import concat
from tensorflow.keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ReduceLROnPlateau
from math import sqrt
from matplotlib import pyplot
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
#from keras.optimizers import adam_v2
from tensorflow.python.keras.optimizer_v2.adam import Adam
#from core.models.lstm_learner import LSTMLearner


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    data = np.reshape(data, newshape=(data.shape[0], 1))
    df = DataFrame(data)
    columns = [df.shift(i) for i in reversed(range(1, lag+1))]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

def build_dataset(dataset, series_size):
    '''
      Receives dataset as a numpy array in the form <time, lat, lon>
    '''
    frame_shp = dataset.shape[1], dataset.shape[2]
    # Define window size
    ws = list([series_size] + list(frame_shp))
    dataset = sliding_window_view(dataset, window_shape=series_size, axis=0)
    dataset = np.swapaxes(dataset, 1, 3)
    # dataset = np.full(shape=(12, 20, 16, 16), fill_value=0)
    #dataset = synthetize_training_dataset(shape = (900, 20, shape[0], shape[1]))

    # Swap the axes representing the number of frames and number of data samples.
    #dataset = np.swapaxes(dataset, 0, 1)

    # We'll pick out 1000 of the 10000 total examples and use those.
    #dataset = dataset[:100, ...]

    # Add a channel dimension since the images are grayscale.
    dataset = np.expand_dims(dataset, axis=-1)

    # Split into train and validation sets using indexing to optimize memory.
    indexes = np.arange(dataset.shape[0])
    #np.random.shuffle(indexes)
    train_index = indexes[: int(0.9 * dataset.shape[0])]
    val_index = indexes[int(0.9 * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    # Normalize the data to the 0-1 range.
    train_dataset = train_dataset / 255
    val_dataset = val_dataset / 255

    # We'll define a helper function to shift the frames, where
    # `x` is frames 0 to n - 1, and `y` is frames 1 to n.
    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :]
        y = data[:, 1 : data.shape[1], :, :]
        return x, y

    # Apply the processing function to the datasets.
    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)

    # Inspect the dataset.
    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

    return x_train, x_val, y_train, y_val, train_dataset, val_dataset


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(ds, scaler=None):
    if scaler is None:
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(ds)

    # transform train
    ds = ds.reshape(ds.shape[0], ds.shape[1])
    ds_scaled = scaler.transform(ds)
    return ds_scaled, scaler

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]



# fit an LSTM network to training data
def fit_lstm(train, batch_size=10, nb_epoch=1, neurons=10, is_stateful=False,
             number_of_hidden_layers=2):
    X, y = train[:, 0:-1], train[:, -1] ###Took a dimension
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(layers.Input(shape=(None, 1)))
    for _ in range(number_of_hidden_layers):
        model.add(layers.LSTM(neurons, stateful=is_stateful,  return_sequences=True))
    model.add(layers.LSTM(neurons, stateful=is_stateful, return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    if not is_stateful:
        model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=True)
    else:
        for i in range(nb_epoch):
            model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
            model.reset_states()
    return model



def get_conv_lstm_model_architecture(train, batch_size=10, nb_epoch=1, neurons=10, 
            is_stateful=False, # todo, 
            number_of_hidden_layers=1, series_size=10, frame_shp=None):
    # ----------- Build CONVLSTM Model -----------

    inp = layers.Input(shape=(None, *frame_shp, 1))

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(x)

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)
    return model

def get_conv_lstm_model_architecture_2(train, batch_size, nb_epoch, neurons, is_stateful=False, # todo
             number_of_hidden_layers=1, series_size=10, frame_shp=None):
    # Define the input shape
    input_shape = (series_size, *frame_shp, 1)

    # Initialize the model
    model = Sequential()

    # Add the ConvLSTM2D layer
    model.add(layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), input_shape=input_shape,
                         padding='same', return_sequences=False))
    return model

def fit_conv_lstm(train, 
            batch_size=10, nb_epoch=1, neurons=10, is_stateful=False, # todo
            number_of_hidden_layers=1, series_size=10):


    X_train, y_train = train[0], train[1]
    frame_shp = X_train.shape[1:3]
    # Define window size
    ws = list(frame_shp)
    ws = tuple([series_size] + ws + [1])
    X = sliding_window_view(X_train, window_shape=ws)
    X = X[:, 0, 0, 0, :, :, :]

    model = get_conv_lstm_model_architecture_2(train, batch_size, nb_epoch, neurons,
                 is_stateful, number_of_hidden_layers, series_size, frame_shp=frame_shp)
    #optimizer = Adam(learning_rate=0.001)
    # optimizer = adam_v2(learning_rate=0.001)

    model.compile(
        optimizer='adam'#, loss='categorical_crossentropy'
    )

    samples, time, lat, long, _ = X.shape
    #X = np.reshape(X, (samples, lat, long, time, 1))
    y = np.reshape(y_train[series_size-1:], (samples, 1, lat, long, 1))

    # Adjust learning rate automatically
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  epsilon=0.0001, patience=1, verbose=1)


    model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=False,
              callbacks=[lr_reduce])

    return model

# make a bath-size-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(batch_size, X.shape[0], 1)
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[:, 0]

def forecast_conv_lstm(model, batch_size, X):
    X = X.reshape(*X.shape[:], 1)
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[..., 0]


def transform_supervised(raw_series, series_size, differentiate=False):
    time_series = raw_series

    # transform data to be stationary
    if differentiate:
        time_series = difference(time_series, 1)

    # transform data to be supervised learning
    supervised = timeseries_to_supervised(time_series, lag=series_size)
    supervised_values = supervised.values

    return supervised_values

def conv_transform_supervised(dataset, series_size):
    # Swap the axes representing the number of frames and number of data samples.
    # dataset = np.swapaxes(dataset, 0, 1)

    # We'll pick out 1000 of the 10000 total examples and use those.
    #dataset = dataset[:100, ...]

    # Add a channel dimension since the images are grayscale.
    dataset = np.expand_dims(dataset, axis=-1)

    # Split into train and validation sets using indexing to optimize memory.
    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[: int(1.0 * dataset.shape[0])]
    val_index = indexes[int(1.0 * dataset.shape[0]):]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    # Normalize the data to the 0-1 range.
    #train_dataset = train_dataset / 255
    #val_dataset = val_dataset / 255

    # We'll define a helper function to shift the frames, where
    # `x` is frames 0 to n - 1, and `y` is frames 1 to n.
    def create_shifted_frames(data):
        x = data[0: data.shape[0] - 1:, :, :, :]
        y = data[1: data.shape[0], :, :, :]
        return x, y

    # Apply the processing function to the datasets.
    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)

    # Inspect the dataset.
    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

    return x_train, x_val, y_train, y_val, train_dataset, val_dataset


def save_model_as_h5(model, model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + ".h5")
    print("Saved model to disk")

def load_model_from_h5(model_directory, model_name):
    # Loads metadata from json file
    json_file = open(model_directory + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # Load weights into new model
    model.load_weights(model_directory + model_name + '.h5')
    # Loads metadata from database
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])







# def train_test_lstm(models_dir, model_name, training_data):
#   retrain_model = True
#   x_train, x_val, y_train, y_val, train_dataset, val_dataset = build_dataset(training_data, series_size=10)  
#   if retrain_model:
#     lstm_model = LstmLearner("", models_dir + model_name, auto_loading=False)
#     lstm_model.update_architecture(neurons=32, nb_epochs=100,
#                                    batch_size=100, number_of_hidden_layers=2)
#     lstm_model.train(train, series_size)

#     np.save(models_dir + model_name + ".npy", train)
#     model_training.save_model_as_h5(lstm_model.get_model(), models_dir + model_name)
#   else:
#     lstm_model = LstmLearner("", models_dir + model_name, auto_loading=True)

#   supervised_test = model_training.transform_supervised(test, series_size)
#   results = lstm_model.predict(lstm_model.get_model(), supervised_test)