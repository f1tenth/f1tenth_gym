import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml
from utils import read_csv

map_id = 1

map_name = "/Users/meraj/workspace/f1tenth_gym/work/tracks"
map_png  = map_name + '/maps/map{}.png'.format(map_id)
map_csv  = map_name + '/maps/map{}.csv'.format(map_id)
map_yaml = map_name + '/centerline/map{}.yaml'.format(map_id)


image = mpimg.imread(map_png)
map_data = np.array(read_csv(map_csv))


with open(map_yaml, 'r') as file:
    yaml_data = yaml.safe_load(file)

map_origin = yaml_data['origin'][0:2]
map_resolution = yaml_data['resolution']


map_x, map_y = (map_data[:, 1] - map_origin[0]) / map_resolution, (map_data[:, 2] - map_origin[1])/ map_resolution


flipped_image = np.flipud(image)
plt.imshow(flipped_image)

plt.plot(map_x, map_y, markersize=1)
plt.show()