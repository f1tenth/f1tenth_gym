
import sys
# sys.path.insert(0, './commonroad-vehicle-models/PYTHON/')

# from vehiclemodels.init_st import init_st

# from vehiclemodels.parameters_vehicle2 import parameters_vehicle2

# from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st

from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid



import shapely.geometry as geom

import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm

from datetime import datetime
from pathlib import Path
from globals import *

import csv  

def solveEuler(func, x0, t, args):
    history = np.empty([len(t), len(x0)])
    history[0] = x0
    
    x = x0
    #Calculate dt vector
    for i in range(1, len(t)):
        x = x + np.multiply(t[i] - t[i-1] ,func(x, t, args[0], args[1]))
        history[i] = x
    return history


'''
Represents the "real car", calculated by the l2race server
'''
class Car:
    def __init__(self, ):

        # initial_position = track.initial_position
        self.state = [0.0,0.,0.,0.,0.,0.,0.]
        self.tEulerStep = 0.01
        self.state_history = [] #Hostory of real car states
        self.control_history = [] #History of controls applied every timestep
        self.tControlSequence =  0.02
        self.params = {
                'mu': 1.0489,       # friction coefficient  [-]
                'C_Sf': 4.718,      # cornering stiffness front [1/rad]
                'C_Sr': 5.4562,     # cornering stiffness rear [1/rad]
                'lf': 0.15875,      # distance from venter of gracity to front axle [m]
                'lr': 0.17145,      # distance from venter of gracity to rear axle [m]
                'h': 0.074,         # center of gravity height of toal mass [m]
                'm': 3.74,          # Total Mass of car [kg]
                'I': 0.04712,       # Moment of inertia for entire mass about z axis  [kgm^2]
                's_min': -0.4189,   # Min steering angle [rad]
                's_max': 0.4189,    # Max steering angle [rad]
                'sv_min': -3.2,     # Min steering velocity [rad/s]
                'sv_max': 3.2,      # Max steering velocity [rad/s]
                'v_switch': 7.319,  # switching velocity [m/s]
                'a_max': 9.51,      # Max acceleration [m/s^2]
                'v_min':-5.0,       # Min velocity [m/s]
                'v_max': 20.0,      # Max velocity [m/s]
                'width': 0.31,      # Width of car [m]
                'length': 0.58      # Length of car [m]
                }

 
    def car_dynamics(self,x, t, u, p):
        """
        Dynamics of the simulated car from common road
        To use other car dynamics than the defailt ones, comment out here
        @param x: The cat's state
        @param t: array of times where the state has to be evaluated
        @param u: Control input that is applied on the car
        @param p: The car's physical parameters
        @returns: the commonroad car dynamics function, that can be integrated for calculating the state evolution
        """
        # f = vehicle_dynamics_ks(x, u, p)
        # f = vehicle_dynamics_st(x, u, p)


        # print("x", x)
        # print("u", u)
        # print("SELF PARAMS", self.params)

        f = vehicle_dynamics_st(
            x,
            u,
            self.params['mu'],
            self.params['C_Sf'],
            self.params['C_Sr'],
            self.params['lf'],
            self.params['lr'],
            self.params['h'],
            self.params['m'],
            self.params['I'],
            self.params['s_min'],
            self.params['s_max'],
            self.params['sv_min'],
            self.params['sv_max'],
            self.params['v_switch'],
            self.params['a_max'],
            self.params['v_min'],
            self.params['v_max'])

        # f = vehicle_dynamics_std(x, u, p)
        # f = vehicle_dynamics_mb(x, u, p)
        # print("f", f)
        return f

   

    def step(self, control_input):
        '''
        Move the car one step due to a given control input
        Is also able to check if the car crossed the finish line of the track and collect lap times
        Draw the car's history 
        @param control_input {control input} 
        '''

        t = np.arange(0, self.tControlSequence, self.tEulerStep) 

        x_next = solveEuler(self.car_dynamics, self.state, t, args=(control_input, self.params))

        self.state = x_next[-1]

        self.state_history.append(x_next)
        self.control_history.append(control_input)


    

    def save_history(self, filename = None):
        '''
        Save all past states of the car into a csv file
        @param filename <string>: Optional: The csv file's name. if left empty the files are saved into the ExperimentRecordings folder with the current datetime as name. 
        '''

        print("Saving history...")
        
        # np.savetxt("racing/last_car_state.csv", self.state, delimiter=",", header="x1,x2,x3,x4,x5,x6,x7")
        
        control_history = np.array(self.control_history)
        np.savetxt("ExperimentRecordings/control_history.csv", control_history, delimiter=",", header="u1,u2")

        header=["time", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "u1", "u2"]
        state_history = np.array(self.state_history)
        state_history = state_history.reshape(state_history.shape[0] * state_history.shape[1],len(self.state))
        np.savetxt("ExperimentRecordings/car_state_history.csv", state_history, delimiter=",", header="x1,x2,x3,x4,x5,x6,x7")


        cut_state_history = state_history[0::20]
        now = datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
    
        file = 'ExperimentRecordings/history-{}.csv'.format(now)
        if filename is not None:
            file = filename

        with open(file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)


            #Meta data
            f.write("# Datetime: {} \n".format(now))
            # f.write("# Track: {} \n".format( self.track.track_name))
            f.write("# Saving: {}\n".format("{}s".format(self.tControlSequence)))
            f.write("# Model: {}\n".format("MPPI, Ground truth model"))

            writer.writerow(header)
            time = 0
            for i in range(len(cut_state_history)):

                state_and_control = np.append(cut_state_history[i],control_history[i])               
                time_state_and_control = np.append(time, state_and_control)
                writer.writerow(time_state_and_control)
                time = round(time+self.tControlSequence, 2)


    
    def draw_history(self, filename = None):
        '''
        Plot all past states of the car into a csv file
        @param filename <string>: Optional: The image's name. if left empty the files are saved into the ExperimentRecordings folder with the current datetime as name. 
        '''
        plt.clf()


        # angles = np.absolute(self.track.AngleNextCheckpointRelative)
        # plt.scatter(self.track.waypoints_x,self.track.waypoints_y, c=angles) #track color depending on angles
        # plt.scatter(self.track.waypoints_x,self.track.waypoints_y, color="#000")

        plt.ylabel('Position History')
        s_x = []
        s_y = []
        velocity = []
    
        for trajectory in self.state_history:
            for state in trajectory:
                s_x.append(state[0])
                s_y.append(state[1])
                velocity.append(state[3]) 

        index = 0
        scatter = plt.scatter(s_x,s_y, c=velocity, cmap = cm.jet)
        index += 1

        colorbar = plt.colorbar(scatter)
        colorbar.set_label('speed')

        now = datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")

        file = 'ExperimentRecordings/history-{}.png'.format(now)
        if filename is not None:
            file = filename

        plt.savefig(file)
    
