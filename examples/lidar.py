import yaml

import time
import gym
import pyglet
from pyglet.gl import glMatrixMode, glLoadIdentity, gluOrtho2D, GL_PROJECTION, GL_MODELVIEW
import numpy as np
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
from waypoint_follow import PurePursuitPlanner
from line_alg import drawLine, _drawPixel

def lerp_gray(value: int) -> int:
    """Maps a value from range [0, 30] to [0, 255] using linear interpolation."""
    return int((value / 30) * 200 + 50)

def clamp(n, smallest, largest): 
    return max(smallest, min(n, largest - 1))

render_mode = "human" # human or human_fast

def setup_orthographic_projection(window):
    """ Sets up an orthographic projection to use pixel-based coordinates. """
    window.switch_to()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, window.width, 0, window.height)  # Bottom-left is (0,0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def get_random_image(width=256, height=256):
    grayscale_data = (np.random.rand(height, width) * 255).astype(np.uint8)
    return grayscale_data

def get_gradient_image(width=256, height=256):
    grayscale_data = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    return grayscale_data

def lidar_scan_to_image(
        scan: list[list[float]],                    # the data. A list of agents' ray lengths.
        agent_ego_id: int=0,                        # the agent id in the data (the index). Default 0
        blinded: bool=True,                         # does the scan stop at the first wall? Default true
        winding_dir: str='CCW',                     # do the rays wind clockwise or counterclockwise. The docs don't say so I've left it up to the user.
        starting_angle=0,                           # in which direction do the rays start? Again the docs don't say so I've left it to the user.
        scaling_factor: float = 10,                  # how much to scale the lengths received in the image.
        bg: str = 'black',                          # black bg with white walls or white bg with black walls.
        draw_center: bool=True,                     # should it draw a dot where the car is?
        output_image_dims: tuple[int]=(256, 256),   # the dimensions of the output image
        num_beams: int=1080                         # the number of beams (the data comes in at 1080)
    ):  
    assert winding_dir in ['CW', 'CCW']
    assert bg in ['black', 'white']

    BLACK = 0
    WHITE = 255
    BG_COLOR = BLACK if bg == 'black' else WHITE
    DRAW_COLOR = WHITE if bg == 'black' else BLACK

    # create the output image
    # Initialize a blank grayscale image
    image = np.ones((output_image_dims[0], output_image_dims[1]), dtype=np.uint8) * BG_COLOR

    # additional data
    data = scan[agent_ego_id]
    data = data[::1080//num_beams] # cut down the size of the data
    # if CW, add. if CCW, subtract
    dir = 1 if winding_dir == 'CCW' else -1
    # calc where the center is
    center = (output_image_dims[0]//2, output_image_dims[1]//2)
    n = len(data)

    # drawing funcs
    def point(x, y):
        if 0 <= x < output_image_dims[0] and 0 <= y < output_image_dims[1]:
            _drawPixel(int(x), int(y), image, DRAW_COLOR)

    def line(x, y, color=DRAW_COLOR):
        drawLine(
            int(center[0]), 
            int(center[1]), 
            clamp(int(x), 0, output_image_dims[0]),
            clamp(int(y), 0, output_image_dims[1]), 
            image, 
            color
        )

    def draw_square(x, y, size=6, color=WHITE):
        """ Draw a square point with given size at (x, y). """
        half_size = size // 2
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                xi, yi = x + dx, y + dy
                if 0 <= xi < 256 and 0 <= yi < 256:
                    image[yi, xi] = color  # Set pixel to color

    if draw_center:
        draw_square(center[0], center[1])
    
    for i, r in enumerate(data):
        theta = starting_angle + dir * 2 * np.pi * i / (n - 1)
        x = round(scaling_factor * r * np.cos(theta) + center[0])
        y = round(scaling_factor * r * np.sin(theta) + center[1])
        line(x, y, lerp_gray(r))
        draw_square(x, y, size=2, color=lerp_gray(r))

    return image

def grayscale_to_rgb(data):
    """
    Takes in single channel grayscale data and returns RGB
    """
    rgb_data = np.stack([data] * 3, axis=-1)  # Shape: (256, 256, 3)
    return rgb_data

def render_image_to_window(data: np.ndarray, window):
    """
    Takes in a single channel, 256x256 grayscale image and draws it to a window.
    """
    window.switch_to()
    window.clear()

    rgb_data = grayscale_to_rgb(data)
    image = pyglet.image.ImageData(256, 256, 'RGB', rgb_data.tobytes())
    image.blit(0, 0, width=256, height=256)

    window.flip()

def main():
    """
    main entry point
    """

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 1.375}#0.90338203837889}
    
    # Create the additional windows
    blinded_window = pyglet.window.Window(256, 256, "Blinded LIDAR")
    nonblinded_window = pyglet.window.Window(256, 256, "Non-blinded LIDAR")
    # set their positions
    blinded_window.set_location(1100, 100)
    nonblinded_window.set_location(1100, 600)
    # and setup their projections
    setup_orthographic_projection(blinded_window)
    setup_orthographic_projection(nonblinded_window)

    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    planner = PurePursuitPlanner(conf, (0.17145+0.15875)) #FlippyPlanner(speed=0.2, flip_every=1, steer=10)

    def render_callback(env_renderer):
        # custom extra drawing function
        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner.render_waypoints(env_renderer)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    env.add_render_callback(render_callback)
    
    obs, step_reward, done, _ = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    # Define the callback to close windows when the main window closes
    def on_close():
        blinded_window.close()
        nonblinded_window.close()
        env.renderer.close()
        pyglet.app.exit()  # Ensures all Pyglet event loops stop
        
    env.renderer.push_handlers(on_close)

    laptime = 0.0
    start = time.time()

    while not done:
        # scanning info
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward

        # lidar func
        blind_scan = lidar_scan_to_image(scan=obs['scans'], blinded=True)
        scan = lidar_scan_to_image(scan=obs['scans'], blinded=True)

        env.render(mode=render_mode)
        render_image_to_window(data=blind_scan, window=blinded_window)
        render_image_to_window(data=scan, window=nonblinded_window)

        # switch back to main window
        env.renderer.switch_to()
        
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

if __name__ == '__main__':
    main()