# The F1TENTH Gym environment

This is the repository of the F1TENTH Gym environment.

This project is still under heavy developement.

## Installation (With Docker)
We recommend using the provided Dockerfile to create containers for this simulation environment. Note that if you need to use the visualizer in docker, you'll need Linux with an NVIDIA GPU.

To build and run the docker containers without the Visualization, note that you might need to run these with ```sudo``` depending on how you setup docker:
```bash
$ ./build_docker.sh
$ ./docker.sh
```

## Installation (Native)
The environment officially supports Python3, Python2 might also work. You'll need several dependencies to run this environment:

### Eigen and protobuf dependencies:

```bash
$ sudo apt-get install -y libzmq3-dev build-essential autoconf libtool libeigen3-dev
$ sudo cp -r /usr/include/eigen3/Eigen /usr/include
```

### Protobuf:

```bash
$ git clone https://github.com/google/protobuf.git
$ cd protobuf
$ ./autogen.sh
$ ./configure
$ make -j4
$ sudo make install
$ ldconfig
$ make clean
```

### Python packages:

```bash
$ pip3 install --user numpy scipy numba zmq pyzmq Pillow gym protobuf pyyaml msgpack==0.6.2
```

### To install the simulation environment natively, clone this repo.

```bash
$ git clone https://github.com/f1tenth/f1tenth_gym
```

### Then install the env via the following steps:
```bash
$ cd f1tenth_gym
$ mkdir build
$ cd build
$ cmake ..
$ make
$ cp sim_requests_pb2.py ../gym/
$ cd ..
$ pip3 install --user -e gym/
```

### Optional Visualizer using Pangolin:

#### Pangolin dependencies:

```bash
$ sudo apt-get install -y libgl1-mesa-dev libglew-dev cmake
$ pip3 install --user pyopengl Pillow pybind11
$ git clone https://github.com/hzheng40/Pangolin
$ cd Pangolin
$ git submodule init && git submodule update
```

#### Building Pangolin
First, you'll have to point cmake to the correct python interp path. To find the paths:

```bash
$ which python3
```

and it should return something like this:
```bash
/usr/bin/python3
```

In the Pangolin directory:

```bash
$ mkdir build
$ cd build
$ cmake -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3 ..
$ cmake --build .
```

Next you'll need to copy the built python library file to any of the directories on your python path using:

```bash
python3 -c "import sys; print('\n'.join(sys.path))"
```

A good choice is the dist-packages directory, it should look something like this:

```bash
/usr/lib/python3/dist-packages
```

Then in the build directory:

```bash
sudo cp src/pypangolin.cpython-36m-x86_64-gnu.so /usr/lib/python3/dist-packages
```

Note that the name of the library file on your system might be different, but it'll be in the ```Pangolin/build/src directory``` after the building is done. For example the filename shown above is from building on a Ubuntu 64 bit system with Python3.6.

## Visualization in Docker

1) Run ./build_docker_ui.sh
Note you may need to use sudo with this command depending on the way you installed docker.

2) If you have docker 19.03 or later, install Nvidia Container Toolkit following the instructions here: (https://github.com/NVIDIA/nvidia-docker), and run ./docker_ui.sh. Note that you may need to use sudo with this command depending on the way you installed docker.

3) If you have an older version of docker and are using nvidia-docker 1.0, run ./docker_ui_nvidia_docker_1.0.sh. Note that you may need to use sudo with this command depending on the way you installed docker.

4) If you have an older version of docker and are using nvidia-docker 2.0, run ./docker_ui_nvidia_docker_2.0.sh. Note that you may need to use sudo with this command depending on the way you installed docker.

## Example Usage
You can step through the environment with the usual step function:
```python
import gym

# making the environment
racecar_env = gym.make('f110_gym:f110-v0')

# loading the map (uses the ROS convention with .yaml and an image file)
map_path = 'your/path/to/the/map/file.yaml'
map_img_ext = '.png' # png extension for example
executable_dir = 'your/path/to/f1tenth_gym/build/'

# loading physical parameters of the car
# These could be identified on your own system
mass= 3.74
l_r = 0.17145
I_z = 0.04712
mu = 0.523
h_cg = 0.074
cs_f = 4.718
cs_r = 5.4562

racecar_env.init_map(map_path, map_img_ext, False, False)
racecar_env.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass, executable_dir, double_finish=True)

# Initial state (for two cars)
initial_x = [0.0, 2.0]
initial_y = [0.0, 0.0]
initial_theta = [0.0, 0.0]
lap_time = 0.0

# Resetting the environment
obs, step_reward, done, info = racecar_env.reset({'x': initial_x,
                                                  'y': initial_y,
                                                  'theta': initial_theta})
# Simulation loop
while not done:

    # Your agent here
    ego_speed, opp_speed, ego_steer, opp_steer = agent.plan(obs)

    # Stepping through the environment
    action = {'ego_idx': 0, 'speed': [ego_speed, opp_speed], 'steer': [ego_steer, opp_steer]}
    obs, step_reward, done, info = racecar_env.step(action)

    # Getting the lap time
    lap_time += step_reward
```
