import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import metadata


# Function to generate time series data for each region with specified pattern
def generate_region_data(num_series, num_time_steps, pattern_func):
    return np.array([
        pattern_func(num_time_steps) + np.random.randn(num_time_steps)
        for _ in range(num_series)
    ])

# Function to generate spatio-temporal dataset
def generate_from_patterns(
    regions, num_series_per_region, num_time_steps, patterns
):
    data = {}

    # Create dict of numpy arrays: 
    # Format: region: <num_series_per_region, num_time_steps> 
    for region in regions:
        region_data = {}
        for pattern_name, pattern_func in patterns.items():
            region_data[pattern_name] = generate_region_data(
                num_series_per_region, num_time_steps, pattern_func
            )
        data[region] = region_data

    # Create Pandas Dataframe
    frames = []
    for region, region_data in data.items():
        for pattern, pattern_data in region_data.items():
            for i in range(num_series_per_region):
                series_name = f"{region}_{pattern}_{i+1}"
                series_data = pd.Series(
                    pattern_data[i], name=series_name
                )
                frames.append(series_data)

    return data, pd.concat(frames, axis=1)

def plot_dataset(spatio_temporal_dataset, regions, patterns):
    # Plot example time series from different regions and patterns
    plt.figure(figsize=(12, 6))
    for i, region in enumerate(regions):
        for j, pattern in enumerate(patterns):
            series_name = f"{region}_{pattern}_1"
            plt.subplot(4, len(patterns), i * len(patterns) + j + 1)
            plt.plot(spatio_temporal_dataset[series_name], label=series_name)
            plt.title(f"{region} - {pattern}")
            plt.legend()

    plt.tight_layout()
    plt.show()

def generate_series_block(pattern_function, block_length, series_size):
    data_3d = np.empty((block_length, block_length, series_size))
    for i in range(block_length):
        for j in range(block_length):
            data_3d[i, j, :] = pattern_function(series_size)    
    return data_3d


def generate_spatio_temporal_dataset():
    series_patterns = metadata.series_patterns
    patterns_frame  = metadata.patterns_frame
    block_length    = metadata.block_length
    time_dim_len     = metadata.time_dimension_len 
    
    data = np.zeros((len(patterns_frame) * block_length, 
            len(patterns_frame[0]) * block_length, time_dim_len))

    for i, _ in enumerate(patterns_frame):
        for j, _ in enumerate(patterns_frame[0]):
            pattern_key = patterns_frame[i][j]
            series_block = generate_series_block(
                series_patterns[pattern_key].function, block_length, time_dim_len)
            i_start, j_start = i * block_length, j * block_length             
            data[i_start:i_start+block_length, j_start:j_start+block_length] = series_block

    # Reorder the dataset to shape (time, lat, lon)
    data = np.transpose(data, (2, 0, 1))
    return data
