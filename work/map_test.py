import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


image = mpimg.imread('/Users/meraj/workspace/f1tenth_gym/examples/example_map.png')

def read_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=';', skip_header=3)
    return data


map_data = read_csv('/Users/meraj/workspace/f1tenth_gym/examples/example_waypoints.csv')
map_data_np = np.array(map_data)

map_x, map_y = map_data_np[:, 1], map_data_np[:, 2]
print(map_data_np[0])

plt.imshow(image, extent=[-80, 23, -47, 58], aspect='auto')

plt.plot(map_x, map_y, markersize=1)
plt.show()