##########################################
########         Constants        ########
##########################################

from os import terminal_size


T_CONTROL = 0.2
T_EULER_STEP = 0.2


##########################################
#######         Experiments        #######
##########################################

SIMULATION_LENGTH = 1000
DRAW_LIVE_HISTORY = False
DRAW_LIVE_ROLLOUTS = False
DRAW_TRACK_SEGMENTS = False
PLOT_LIDAT_DATA = False
PATH_TO_EXPERIMENT_RECORDINGS = "./ExperimentRecordings"


##########################################
#########       Car & Track     ##########
##########################################

INITIAL_SPEED = 11
CONTINUE_FROM_LAST_STATE = False
ALWAYS_SAVE_LAST_STATE = False
EXIT_AFTER_ONE_LAP = False
COLLECT_LAP_TIMES = True

TRACK_NAME = "track_2"
M_TO_PIXEL = 0.1
TRACK_WIDTH = 100


##########################################
####   Neural MPC Car Controller     #####
##########################################

# Path Prediction
CONTROLLER_PREDICTIOR = "nn"
# CONTROLLER_MODEL_NAME = "Dense-128-128-128-128-invariant-10" # Accurate
CONTROLLER_MODEL_NAME = "Dense-64-64-64" # Fast
# CONTROLLER_MODEL_NAME = "Dense-128-128-128-128-small" # Small training data


# Initializing parameters
NUMBER_OF_INITIAL_TRAJECTORIES = 200
INITIAL_STEERING_VARIANCE = 2

INITIAL_ACCELERATION_MEAN = 1
INITIAL_ACCELERATION_VARIANCE = 0


# Parameters for rollout
NUMBER_OF_TRAJECTORIES = 50
STEP_STEERING_VARIANCE = 1
STEP_ACCELERATION_VARIANCE = 0
NUMBER_OF_STEPS_PER_TRAJECTORY = 15
INVERSE_TEMP = 5

# Relation to track
NUMBER_OF_NEXT_WAYPOINTS = 15
NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS = 2
ANGLE_COST_INDEX_START = 5
ANGLE_COST_INDEX_STOP = 15

# Relations to car
MAX_SPEED = 15
MAX_COST = 10000


##########################################
#########       NN Training     ##########
##########################################

# Artificial data generation
# The training data is saved/retreived in nn_prediction/training/data/[filename]
DATA_GENERATION_FILE = "training_data_10-30_600x100x10.csv"

# Training parameters
MODEL_NAME = "Dense-64-64-64"
TRAINING_DATA_FILE = "training_data_10-30_600x100x10.csv"
NUMBER_OF_EPOCHS = 40
BATCH_SIZE = 64
PREDICT_DELTA = True
NORMALITE_DATA = True
CUT_INVARIANTS = True
