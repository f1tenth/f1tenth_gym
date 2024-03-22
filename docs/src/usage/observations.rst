.. _observations:

Observations
=====================

For the environment, several **types of observations** can be used.

- `original` : Original observation from the old simulator. This is default for compatibility.
- `kinematic_state` : Kinematic state observation, which includes `pose_x, pose_y, delta, linear_vel_x, pose_theta`.
- `dynamic_state` : Dynamic state observation, which includes `pose_x, pose_y, delta, linear_vel_x, pose_theta, ang_vel_z, beta`.
- `features` : Customisable observation, which includes all the features defined in the `features` argument.

**Note:** All the agents will have the same observation type.

Observations Configuration
--------------------------
Each environment comes with a *default* observation,
which can be changed or customised using environment configurations.

Observations can be configured at the environment creation:

.. code:: python

    import gymnasium as gym

    env = gym.make(
		"f110_gym:f110-v0",
		config={
			"observation_config": {
				"type": "features",
				"features": ["linear_vel_x", "scan"]
			},
		})
    obs, infos = env.reset()
or after the environment creation:

.. code:: python

    import gymnasium as gym

    env = gym.make("f110_gym:f110-v0")
    env.configure({
		"observation_config": {
			"type": "features",
			"features": ["linear_vel_x", "scan"]}
		})
    obs, infos = env.reset()