import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


image = mpimg.imread('/Users/meraj/workspace/f1tenth_gym/work/tracks/maps/map6.png')

def read_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=';', skip_header=3)
    return data


map_data = read_csv('/Users/meraj/workspace/f1tenth_gym/work/tracks/centerline/map6.csv')
map_data_np = np.array(map_data)

map_x, map_y = (map_data_np[:, 1] + 54.176041943946956) / 0.062500, (map_data_np[:, 2] + 48.35154856753476)/ 0.062500
print(map_data_np[0])


# Create a flipped version of the image
flipped_image = np.flipud(image)

# Show the flipped image
plt.imshow(flipped_image)
# plt.show()


# plt.imshow(image)

plt.plot(map_x, map_y, markersize=1)
plt.show()