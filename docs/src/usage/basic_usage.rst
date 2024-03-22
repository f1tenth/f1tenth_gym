.. _basic_usage:

Basic example
=====================

The environment can work out of the box without too much customization.

A gym env could be instantiated without any extra arguments using the ``gym.make()`` function.
By default, it spawns two agents in the Spielberg racetrack.

The simulation can be reset by calling the ``reset()`` method
and step forward in time by calling the ``step()`` method with a joint action for all agents.

A working example can be found in ``examples/waypoint_follow.py``.


Usage
-----

.. code:: python

    import gymnasium as gym

    env = gym.make("f110_gym:f110-v0", render_mode="human")
    obs, infos = env.reset()

    while not done:
        # sample random action
        actions = env.action_space.sample()
        # step simulation
        obs, step_reward, done, info = racecar_env.step(actions)
        env.render()

Default configuration
---------------------

.. code:: python

    {
	"seed": 12345,
	"map": "Spielberg",
	"params": {
		"mu": 1.0489,
		"C_Sf": 4.718,
		"C_Sr": 5.4562,
		"lf": 0.15875,
		"lr": 0.17145,
		"h": 0.074,
		"m": 3.74,
		"I": 0.04712,
		"s_min": -0.4189,
		"s_max": 0.4189,
		"sv_min": -3.2,
		"sv_max": 3.2,
		"v_switch": 7.319,
		"a_max": 9.51,
		"v_min": -5.0,
		"v_max": 20.0,
		"width": 0.31,
		"length": 0.58,
	},
	"num_agents": 2,
	"timestep": 0.01,
	"ego_idx": 0,
	"integrator": "rk4",
	"model": "st",
	"control_input": ["speed", "steering_angle"],
	"observation_config": {"type": None},
	"reset_config": {"type": None},
    }
