import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.patches as patches


path = "../resources/data/CFSR-2015.nc"
dataset = nc.Dataset(path)
array = dataset.variables['TMP_L100'][0]

plt.xlabel('Lat')
plt.ylabel('Lon')

#highlight_area = patches.Rectangle((60, 62), 20, 22, linewidth=1, edgecolor='r', facecolor='none')
highlight_area = patches.Rectangle((20, 70), 15, 32, linewidth=1, edgecolor='r', facecolor='none')
plt.gca().add_patch(highlight_area)

plt.imshow(array, cmap='viridis') 
plt.colorbar(label='Valor')
plt.savefig('2.jpg', bbox_inches='tight', pad_inches=0)
plt.show()
