import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List of image file names
image_files = ["dtw.jpg"]

methods = ["dtw"]

# Create a figure with four subplots
fig, axes = plt.subplots(2, 2, hspace=0.2)

# Loop through image files and display them in subplots
for i, ax in enumerate(axes.ravel()):
    img = mpimg.imread(image_files[i])
    ax.imshow(img)
    ax.set_title(f"{methods[i]}")

# Adjust layout and display
plt.tight_layout()
plt.show()