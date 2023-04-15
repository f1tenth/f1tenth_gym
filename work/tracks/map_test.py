import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


image = mpimg.imread('/Users/meraj/workspace/f1tenth_gym/work/maps/map_00/racing_map.png')

def read_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=';', skip_header=3)
    return data


map_data = read_csv('/Users/meraj/workspace/f1tenth_gym/work/maps/map_00/racing_map.csv')
map_data_np = np.array(map_data)

map_x, map_y = (map_data_np[:, 1] + 78.21853769831466) / 0.062500, (map_data_np[:, 2] + 44.37590462453829)/ 0.062500
print(map_data_np[0])


# Create a flipped version of the image
flipped_image = np.flipud(image)

# Show the flipped image
plt.imshow(flipped_image)
# plt.show()


# plt.imshow(image)

plt.plot(map_x, map_y, markersize=1)
plt.show()