.. _rendering:

Rendering
=====================

The environment rendering is done with `pygame`.

As by default in gymnasium environments,
there are multiple rendering modes available:

- `human`: for real-time rendering with `pygame`;

- `rgb_array`: for collecting the current frame as a numpy array;

- `rgb_array_list`: for collecting multiple frames into a list. This is useful to generate smooth videos when using frame-skipping.

Additionally, for fast rendering and debugging, we add:

- `human-fast`: faster-than-real-time rendering. Up to 10x faster based on the host machine.

Rendering Configuration
------------------------

The user can customize the rendering mode with the `rendering_mode` parameter in the environment configuration.

.. code:: python

    import gymnasium as gym

    env = gym.make(
        "f110_gym:f110-v0",
        render_mode="human-fast",
    )
    obs, infos = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, infos = env.step(action)
        env.render()


.. note::
    When rendering in `human` mode,
    the user can interact to change the agent to focus on by pressing the left/right/middle mouse button,
    as described by the instructions diplayed in the top of the window.

Parameter Configuration
-----------------------

The user can customize the rendering parameters in the `f1tenth_gym.rendering/rendering.yaml` file:

- `window_size`: width and height of the window in pixels

- `focus_on`: agent id to focus on (e.g., `agent_0`) or `null` for map view

- `zoom_in_factor`: zoom in factor. 1.0 is no zoom, >1.0 is zoom in, <1.0 is zoom out

- `show_wheels`: it toggles the visualization of the vehicle wheels

- `car_tickness`: thickness of the car border

- `show_info`: it toggles the visualization of the instruction text

- `vehicle_palette`: cyclic list of colors for the vehicles (e.g., first agent takes the first color, second agent takes the second color, etc. and then it starts again from the first color)



