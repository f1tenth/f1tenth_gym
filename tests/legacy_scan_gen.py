import numpy as np

import gymnasium as gym


# init
params = {
    "mass": 3.74,
    "l_r": 0.17145,
    "I_z": 0.04712,
    "mu": 0.523,
    "h_cg": 0.074,
    "cs_f": 4.718,
    "cs_r": 5.4562,
}

# test poses
num_test = 10
test_poses = np.zeros((num_test, 3))
test_poses[:, 2] = np.linspace(-1.0, 1.0, num=num_test)

all_scans = {}  # map 1: vegas
for map_name in ["Spielberg", "Monza", "Austin"]:
    print(f"Generating scan data for {map_name}")

    env = gym.make(
        "f110_gym:f110-v0",
        config={
            "map": map_name,
            "num_agents": 1,
            "params": params,
        },
    )

    scan = np.empty((num_test, 1080))
    for i in range(test_poses.shape[0]):
        obs, _ = env.reset(options={"poses": test_poses[i, :][None]})
        agent_id = list(obs.keys())[0]
        scan[i, :] = obs["scans"][0]

    all_scans[map_name] = scan

# package data
np.savez_compressed("legacy_scan.npz", **all_scans)
print("Data saved to legacy_scan.npz")
