import os, sys
from os.path import exists
import fnmatch

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list_all_sub_directories(path):
    return [x[0] for x in os.walk(path)]

def list_all_directories(path):
    return next(os.walk(path))[1]

def get_names_of_models_in_dir(models_path):
    models = fnmatch.filter(os.listdir(models_path), '*.h5')
    for i, m in enumerate(models):
        models[i] = m.split('.h5')[0]
    return models

def list_all_files_in_dir(path, extension=''):
    res = []
    for file in os.listdir(path):
        if file.endswith(extension) or file.endswith('.' + extension):
            res.append(file)
    return(res)

def file_exists(path_to_file):
    return exists(path_to_file)

def get_file_name_from_path(full_path_to_file: str):
    return full_path_to_file.split("/")[-1]

def print_array(array_list):
    from matplotlib import pyplot as plt

    if isinstance(array_list, list):
        n = len(array_list)
        f, axis = plt.subplots(1, n)
        title_list = ["Clustering", "Tiling"]
        for array, i, title in zip(array_list, range(n), title_list):
            axis[i].set_title(title)
            #axis[i].imshow(array, cmap="tab20", vmin=0, vmax=array_list[i].max())
            axis[i].imshow(array, cmap="viridis", vmin=0, vmax=array.max())
    else:
        plt.imshow(array_list, cmap='gray', vmin=0, vmax=array_list[1].max())#, interpolation='nearest')
    plt.show()

def supress_log_messages(supress:bool):
        global original_stdout
        if supress:
            original_stdout = sys.stdout
            null_device = open(os.devnull, 'w')
            sys.stdout = null_device
        else:
            sys.stdout = original_stdout