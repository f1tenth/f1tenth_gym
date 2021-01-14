Basic Usage Example
=====================

The environment can work out of the box without too much customization like this:

.. code:: python

    import gym
    import numpy as np

    # making the environment
    racecar_env = gym.make('f110_gym:f110-v0')
    obs, step_reward, done, info = racecar_env.reset(np.array([[0., 0., 0.], # pose of ego
                                                               [2., 0., 0.]])) # pose of 2nd agent

    # simulation loop
    lap_time = 0.
    while not done:
        # some agent policy that you created
        actions = planner.plan(obs) # numpy.ndarray (num_agents, 2), columns are steering angle and then velocity

        # stepping through the environment
        obs, step_reward, done, info = racecar_env.step(actions)

        lap_time += step_reward