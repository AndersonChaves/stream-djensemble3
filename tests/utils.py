import numpy as np

def create_block_time_series_dataset(dim):
    series_block_1 = np.array(
        ([[[20] * 5] * dim] + [[[25] * 5] * dim]) * dim
    )
    series_block_2 = np.zeros((10, dim, dim))

    series_block_3 = np.array(
        [[[x]*dim]*dim for x in range(-9, 1)]
    )
    
    series_block_4 = np.zeros((10, dim, dim))
    
    row_1 = np.concatenate(
        (series_block_1, series_block_2), axis=1
    )
    row_2 = np.concatenate(
        (series_block_3, series_block_4), axis=1
    )
    
    return np.concatenate((row_1, row_2), axis=2)


def create_noise_time_series_dataset(shape, noise=0):
    time, lat, long = shape

    series = []
    for i in range(lat // 3 * long):
        series.append([k for k in range(time)])
    for i in range(lat // 3 * long):
        series.append([3 - (k % 3) for k in range(time)])
    for i in range((lat // 3 + lat % 3) * long):
        series.append([10 for _ in range(time)])

    array = np.reshape(np.array(series), (lat, long, time))
    array = np.swapaxes(array, 0, 2)

    if noise > 0:
        noise = np.random.normal(0, noise, array.shape)
        array = array + noise
    return array

