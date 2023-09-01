import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from core.utils import create_directory_if_not_exists

def print_array_screen(array_list, title_list = None, cmap_list=None):
    if isinstance(array_list, list):
        n = len(array_list)
        f, axis = plt.subplots(1, n)
        if title_list is None:
            title_list = ["Clustering", "Tiling"]
        if cmap_list is None:
            cmap_list = ["viridis", "viridis", "gray"]
        for array, i, title in zip(array_list, range(n), title_list):
            axis[i].set_title(title)
            #axis[i].imshow(array, cmap="tab20", vmin=0, vmax=array_list[i].max())
            axis[i].imshow(array, cmap=cmap_list[i], vmin=0, vmax=array.max())
    else:
        #plt.imshow(array_list, cmap='gray', vmin=0, vmax=array_list[1].max())#, interpolation='nearest')
        plt.imshow(array_list, cmap='viridis', vmin=0, vmax=array_list[1].max())#, interpolation='nearest')
    plt.show()

def save_figure_from_matrix(matrix : np.array, parent_directory: str, 
                            title: str, write_values=False,font_size=5):
    create_directory_if_not_exists(parent_directory)
    matplotlib.image.imsave(parent_directory + title + ".jpg", matrix)    
    return

def plot_graph_line(error_history, title: str,
                   parent_directory='', write_values=False):
    matplotlib.use('TkAgg')

    fig, ax = plt.subplots()
    print("Error History: ", error_history)
    ax.plot(list(range(len(error_history))), error_history)
    fig.savefig(parent_directory + title + '.png', dpi=160)

    #plt.plot(list(range(len(error_history))), error_history)
    #plt.savefig(parent_directory + title + '.png', dpi=40)

def save_figure_as_scatter_plot(x, y, clusters, title:str, parent_directory='', annotate=False):
    matplotlib.use('TkAgg')
    figure(figsize=(8, 6), dpi=130)
    matplotlib.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=clusters, s=30, cmap='viridis')

    if annotate:
        for i, point in enumerate(zip(x[-25:], y[-25:])):
            ax.annotate(i, (point[0], point[1]))

    fig.savefig(parent_directory + title + '.png')
    plt.close("all")

if __name__ == '__main__':
    matrix = np.random.random((128, 128))
    #matrix = np.random.random((10, 10))
    save_figure_from_matrix(matrix, "../figures/last-buffer", parent_directory='', write_values=False)
    print("Done")