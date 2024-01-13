import numpy as np
import matplotlib
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
#from core.utils import create_directory_if_not_exists
import seaborn as sns

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


def save_array_list_as_image(array_list, file_name, title_list = None, cmap_list=None):
    #create_directory_if_not_exists(parent_directory)

    
    f, axes = plt.subplots(1, len(array_list), figsize=(12, 6))
    if len(array_list) == 1:
        axes = [axes]
    for i, ar in enumerate(array_list):
        if array_list[0].shape[0] < 10:
            sns.set(font_scale=1.0)
        else:
            sns.set(font_scale=0.3)
        sns.heatmap(ar, annot=True, fmt='.1f', 
            cmap='YlGnBu', cbar=False, square=True, ax=axes[i])
        
        sns.set(font_scale=1.0)
        axes[i].set_xlabel('Lat', fontsize=12)
        axes[i].set_ylabel('Lon', fontsize=12)
        if title_list is not None:
            axes[i].set_title(title_list[i])
    plt.tight_layout()

    # Save the figure to a file (e.g., PNG)
    plt.savefig(file_name)

    # Optionally, close the figure to release resources
    plt.close()


#ar = np.array([list(range(3)) for _ in range(3)])
#save_array_list_as_image([ar, ar], "output.jpg", ["tiling", "tiling2"])
#print("Teste")


def save_figure_from_matrix(matrix : np.array, parent_directory: str, 
                            title: str, write_values=False,font_size=5):
    create_directory_if_not_exists(parent_directory)
    matplotlib.image.imsave(parent_directory + title + ".jpg", matrix)    
    return


    img = Image.fromarray(matrix, "RGB") 
    img.save(parent_directory + title + ".jpg")
    return 
    plt.matshow(matrix, cmap=plt.cm.Blues)
    #plt.imshow(matrix)
    #cmap=plt.cm.Blues
    #plt.colorbar()
    plt.savefig(parent_directory + title)
    plt.close("all")
    return

    
    
    matplotlib.use('TkAgg')
    if write_values:
        matplotlib.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots()


    if write_values:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                c = round(matrix[i, j], 1)
                ax.text(i+0.5, j+0.5, str(c), va='center', ha='center')
                # print("I, J: ", i, ' ', j)

    plt.matshow(matrix, cmap=plt.cm.Blues)
    min_val_lin, max_val_lin = 0, matrix.shape[0]
    min_val_col, max_val_col = 0, matrix.shape[1]

    ax.set_xlim(min_val_lin, max_val_lin)
    ax.set_ylim(min_val_col, max_val_col)
    ax.set_xticks(np.arange(max_val_lin))
    ax.set_yticks(np.arange(max_val_col))

    fig.tight_layout()
    ax.grid()
    fig.savefig(parent_directory + title + '.png', dpi=160)
    #plt.pause(1)
    plt.close("all")

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