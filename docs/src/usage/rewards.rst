.. _rewards:

Rewards
=====================

We define a simple reward to encourage the agent to stay alive, giving a reward of `timestep` at each timestep.

.. note::
    The choice of an appropriate reward is a challenging problem, beyond the scope of this project.
    To avoid over-specifying every aspect of the expected racing behaviour in the reward
    (*e.g., safe distance, speed, minimising jerk*),
    we preferred to specify the simplest reward function possible.
    This allows the user to experiment with different reward functions based on their specific requirements.


Custom Reward Function
------------------------------

For custom reward functions, we invite the user to use
`gymnasium.wrappers <https://gymnasium.farama.org/tutorials/gymnasium_basics/implementing_custom_wrappers/>`_.

For example, the following code snippet shows how to implement a custom reward function that
linearly combines a speed reward and a crash penalty:

.. code:: python

    class RacingRewardWrapper(Wrapper):

    def __init__(
        self,
        env: gymnasium.Env,
        agent_id: str,
        reward_speed_weight: float,
        reward_crash_weight: float,
    ):
        super().__init__(env)
        self.agent_id = agent_id
        self.reward_speed_weight = reward_speed_weight
        self.reward_crash_weight = reward_crash_weight

        # sanity checks on the observation space
        if agent_id not in env.observation_space.spaces:
            raise ValueError(f"Agent {agent_id} not found in observation space")
        for feature in ["linear_vel_x", "collision"]:
            if feature not in env.observation_space.spaces[agent_id].spaces:
                raise ValueError(f"{feature} not found in observation space")

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        speed = obs[self.agent_id]["linear_vel_x"]
        crash = obs[self.agent_id]["collision"]
        reward = self.reward_speed_weight * speed + self.reward_crash_weight * crash
        return obs, reward, terminated, truncated, info


The above wrapper can be used on the original environment as follows:

.. code:: python

    env = gymnasium.make(
        "f110_gym:f110-v0",
        config={
            "num_agents": 1,
            "observation_config": {
                "type": "features",
                "features": ["scan", "linear_vel_x", "collision"],
            },
        },
        render_mode="human",
    )
    env = RacingRewardWrapper(
        env, agent_id="agent_0", reward_speed_weight=1.0, reward_crash_weight=-1.0
    )

    obs, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print("step reward: ", reward)
        env.render()

