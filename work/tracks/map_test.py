import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('/Users/meraj/workspace/f1tenth_gym/work/tracks/maps/map1.png')

def read_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=';', skip_header=3)
    return data


map_data = read_csv('/Users/meraj/workspace/f1tenth_gym/work/tracks/centerline/map1.csv')
map_data_np = np.array(map_data)

map_x, map_y = (map_data_np[:, 1] + 78.09318169109325) / 0.062500, (map_data_np[:, 2] + 44.338090443224495)/ 0.062500
print(map_data_np[0])


# Create a flipped version of the image
flipped_image = np.flipud(image)

# Show the flipped image
plt.imshow(flipped_image)

plt.plot(map_x, map_y, markersize=1)
plt.show()