Getting Started
===============

Making an environment
---------------------

This is a quick example of how to make a simple environment:

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

Configuration
-------------

Here is the list of configurable options for the environment:
- Track
- Observations
- Actions
- Rewards
- Dynamics
- Rendering
- Extra tools

.. toctree::
  :caption: Getting started
  :maxdepth: 1

  basic_usage
  observations
  actions
  rewards
  dynamics
  customized_usage
  rendering