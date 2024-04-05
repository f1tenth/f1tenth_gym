.. _actions:

Actions
=====================

Several **types of actions** for longitudinal and lateral control are available in every track.

Lateral actions:

- `steering_angle`: the action sets the target steering angle of the vehicle in `rad`, which is then converted to steering speed.

- `steering_speed`: the action directly sets the steering speed of the vehicle in `rad/s`.

Longitudinal actions:

- `speed`: the action sets the target speed of the vehicle in `m/s`, which is then converted to acceleration.

- `accl`: the action directly sets the vehicle acceleration in `m/s^2`.

**Note:** All the agents will have the same observation type.

Actions Configuration
---------------------
The environment comes with a *default* action type, which can be changed using the environment configuration.

Actions can be configured at the environment creation:

.. code:: python

    import gymnasium as gym

    env = gym.make(
		"f110_gym:f110-v0",
		config={
			"control_input": ["accl", "steering_speed"]
		})
    obs, infos = env.reset()

or after the environment creation:

.. code:: python

    import gymnasium as gym

    env = gym.make("f110_gym:f110-v0")
    env.configure({
        "control_input": ["speed", "steering_angle"]
    })
    obs, infos = env.reset()