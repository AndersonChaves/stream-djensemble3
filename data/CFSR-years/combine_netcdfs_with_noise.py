import os
import xarray as xr

def combine_multiple_netcdf(input_dir, output_file):
    # Get a list of all NetCDF files in the input directory
    netcdf_files = [file for file in os.listdir(input_dir) if file.endswith('.nc')]

    # Create an empty list to store the datasets from all NetCDF files
    datasets = []

    # Open each NetCDF file and append its dataset to the list
    for file in netcdf_files:
        file_path = os.path.join(input_dir, file)
        dataset = xr.open_dataset(file_path)
        datasets.append(dataset)

    # Combine the datasets along the desired dimension (e.g., time)
    combined_dataset = xr.concat(datasets, dim='time')

    # Save the combined dataset to a new NetCDF file
    combined_dataset.to_netcdf(output_file)

    # Close all datasets to release resources
    for dataset in datasets:
        dataset.close()

if __name__ == '__main__':
    input_directory = '.'  # Replace with the path to the directory containing the NetCDF files
    output_file = 'output_combined.nc'  # Replace with the desired output filename

    combine_multiple_netcdf(input_directory, output_file)
