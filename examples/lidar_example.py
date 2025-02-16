import yaml
import time
import gym
import pyglet
from pyglet.gl import glMatrixMode, glLoadIdentity, gluOrtho2D, GL_PROJECTION, GL_MODELVIEW
import numpy as np
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
from waypoint_follow import PurePursuitPlanner
from weap_util.lidar import lidar_to_bitmap

render_mode = "human_fast" # human or human_fast
fov = 2 * np.pi

def setup_orthographic_projection(window):
    """ Sets up an orthographic projection to use pixel-based coordinates. """
    window.switch_to()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, window.width, 0, window.height)  # Bottom-left is (0,0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def render_image_to_window(data: np.ndarray, window):
    """
    Takes in a single channel, 256x256 grayscale image and draws it to a window.
    """
    window.switch_to()
    window.clear()

    image = pyglet.image.ImageData(256, 256, 'RGB', data.tobytes())
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

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4, fov=fov)
    env.add_render_callback(render_callback)
    
    global done
    obs, step_reward, done, _ = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    # Define the callback to close windows when the main window closes
    def on_close():
        global done
        done = True
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
        blind_scan = lidar_to_bitmap(scan=obs['scans'][0], channels=3, fov=fov)
        scan = lidar_to_bitmap(scan=obs['scans'][0], channels=3, fov=fov)

        env.render(mode=render_mode)
        render_image_to_window(data=blind_scan, window=blinded_window)
        render_image_to_window(data=scan, window=nonblinded_window)

        # switch back to main window
        env.renderer.switch_to()
        
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

if __name__ == '__main__':
    main()