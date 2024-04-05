.. _dynamics:

Vehicle Dynamics
=====================

The vehicle dynamics are modeled using a single-track model, as in [AlKM17]_.

We support two vehicle dynamics models:
- Kinematic Single-Track Model (`ks`): Simpler model that considers only the kinematics of the vehicle, i.e., the position, orientation, and velocity, ignoring the forces that act on the vehicle.
- Single-Track Model (`st`): More complex model that considers the forces that act on the vehicle, such as the tire forces.

Despite the fact that the single-track model is a simplified model, it is able to capture the
tire slip effects on the slip angle, which is important for accurate simulation at the physical
limits of the vehicle.


Dynamics Configuration
----------------------
The environment comes with a *default* dynamics model, which can be changed using the environment configuration.

The dynamics model can be configured at the environment creation:

.. code:: python

    import gymnasium as gym

    env = gym.make(
		"f110_gym:f110-v0",
		config={
			"model": "ks"
		})
    obs, infos = env.reset()

or after the environment creation:

.. code:: python

    import gymnasium as gym

    env = gym.make("f110_gym:f110-v0")
    env.configure({
        "model": "ks",
    })
    obs, infos = env.reset()



.. rubric:: References

.. [AlKM17] Althoff, M.; Koschi, M.; Manzinger, S..: CommonRoad: Composable benchmarks for motion planning on roads. In: 2017 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2017.