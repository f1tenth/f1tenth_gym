# Dependencies

# The Pipeline

![pipeline](https://raw.githubusercontent.com/mlab-upenn/racecar_simulator_standalone/billy-dev/gym/media/parallel_sim_figures.png?token=ADBVNYE5Q47XAXYEXL2ULAK5OFO3O "pipeline")

The task worker receives a trajectory, and speed at each waypoint, and parameters, creates the env instance, which creates the sim instance, and calculates a score to send back. Pure pursuit calculation based on the observation should be in gym env (i.e. inside the step function?)