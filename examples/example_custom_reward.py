import gymnasium
from gymnasium import Wrapper


class RacingRewardWrapper(Wrapper):
    """
    Example of custom reward which encourage speed and penalize crashes.

    reward(state) = w_{speed} * state_{speed} + w_{crash} * state_{crash}
    """

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

tot_reward = 0.0
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    print("step reward: ", reward)
    tot_reward += reward

    env.render()

print("total reward: ", tot_reward)
env.close()
