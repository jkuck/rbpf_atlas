import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.monte_carlo import stratified_resample
import filterpy
#import matplotlib.pyplot as plt
#import matplotlib.cm as cmx
#import matplotlib.colors as colors
from scipy.stats import multivariate_normal
from scipy.stats import gamma
from scipy.special import gdtrc
import random
import copy 
import math
from numpy.linalg import inv
import pickle
import sys
import resource
import errno
from munkres import Munkres
from collections import deque

#sys.path.insert(0, "/Users/jkuck/rotation3/clearmetrics")
#import clearmetrics
sys.path.insert(0, "./KITTI_helpers")
from learn_params1 import get_meas_target_set
from learn_params1 import get_meas_target_sets_lsvm_and_regionlets
from learn_params1 import get_meas_target_sets_regionlets_general_format
from learn_params1 import get_meas_target_sets_mscnn_general_format
from learn_params1 import get_meas_target_sets_mscnn_and_regionlets

from jdk_helper_evaluate_results import eval_results

DATA_PATH = "/atlas/u/jkuck/rbpf_target_tracking/KITTI_helpers/data"


#from multiple_meas_per_time_assoc_priors import HiddenState
#from proposal2_helper import possible_measurement_target_associations
#from proposal2_helper import memoized_birth_clutter_prior
#from proposal2_helper import sample_birth_clutter_counts
#from proposal2_helper import sample_target_deaths_proposal2
random.seed(5)
np.random.seed(seed=5)
#MKL_NUM_THREADS=1
#NUMEXPR_NUM_THREADS=1

InfLoopDEBUG = False
InfLoopDEBUG1 = False
InfLoopDEBUG2 = False
InfLoopDEBUG3 = False

import cProfile
import time
import os
from run_experiment_batch import DIRECTORY_OF_ALL_RESULTS
from run_experiment_batch import CUR_EXPERIMENT_BATCH_NAME
from run_experiment_batch import SEQUENCES_TO_PROCESS
from run_experiment_batch import get_description_of_run


USE_CREATE_CHILD = True #speed up copying during resampling
RUN_ONLINE = True #save online results 
#near online mode wait this many frames before picking max weight particle 
ONLINE_DELAY = 3
#Write results of the particle with the largest importance
#weight times current likelihood, double check doing this correctly
FIND_MAX_IMPRT_TIMES_LIKELIHOOD = False 
#if true only update a target with at most one measurement
#(i.e. not regionlets and then lsvm)
MAX_1_MEAS_UPDATE = True

######DIRECTORY_OF_ALL_RESULTS = '/atlas/u/jkuck/rbpf_target_tracking'
######CUR_EXPERIMENT_BATCH_NAME = 'test_copy_correctness_orig_copy'
#######run on these sequences
#######SEQUENCES_TO_PROCESS = [0]
######SEQUENCES_TO_PROCESS = [i for i in range(21)]

#Variables defined in main ARE global I think, not needed here (triple check...)
#define global variables, which will be set in main
#SCORE_INTERVALS = None
#TARGET_EMISSION_PROBS = None
#CLUTTER_PROBABILITIES = None
#BIRTH_PROBABILITIES = None
#MEAS_NOISE_COVS = None
#BORDER_DEATH_PROBABILITIES = None
#NOT_BORDER_DEATH_PROBABILITIES = None


#SEQUENCES_TO_PROCESS = [i for i in range(21)]
#eval_results('./rbpf_KITTI_results', SEQUENCES_TO_PROCESS)
#sleep(5)
#RBPF algorithmic paramters

RESAMPLE_RATIO = 2.0 #resample when get_eff_num_particles < N_PARTICLES/RESAMPLE_RATIO

DEBUG = False

USE_PYTHON_GAUSSIAN = False #if False bug, using R_default instead of S, check USE_CONSTANT_R

#default time between succesive measurement time instances (in seconds)
default_time_step = .1 

USE_CONSTANT_R = True
#For testing why score interval for R are slow
CACHED_LIKELIHOODS = 0
NOT_CACHED_LIKELIHOODS = 0




#from learn_params
#BIRTH_COUNT_PRIOR = [0.9371030016191306, 0.0528085689376012, 0.007223813675426578, 0.0016191306513887158, 0.000747291069871715, 0.00012454851164528583, 0, 0.00012454851164528583, 0.00012454851164528583, 0, 0, 0, 0, 0.00012454851164528583]
#from learn_params1, not counting 'ignored' ground truth
BIRTH_COUNT_PRIOR = [0.95640802092415, 0.039357329679910326, 0.0027400672561962883, 0.0008718395815170009, 0.00012454851164528583, 0.00012454851164528583, 0, 0.00024909702329057166, 0, 0, 0.00012454851164528583]

def get_score_index(score_intervals, score):
	"""
	Inputs:
	- score_intervals: a list specifying detection score ranges for which parameters have been specified
	- score: the score of a detection

	Output:
	- index: output the 0 indexed score interval this score falls into
	"""

	index = 0
	for i in range(1, len(score_intervals)):
		if(score > score_intervals[i]):
			index += 1
		else:
			break
	assert(score > score_intervals[index]), (score, score_intervals[index], score_intervals[index+1]) 
	if(index < len(score_intervals) - 1):
		assert(score <= score_intervals[index+1]), (score, score_intervals[index], score_intervals[index+1])
	return index


#regionlet detection with score > 2.0:
#from learn_params
#P_TARGET_EMISSION = 0.813482 
#from learn_params1, not counting 'ignored' ground truth
P_TARGET_EMISSION = 0.813358070501
#death probabiltiies, for sampling AFTER associations, conditioned on un-association
#DEATH_PROBABILITIES = [-99, 0.1558803061934586, 0.24179829890643986, 0.1600831600831601, 0.10416666666666667, 0.08835341365461848, 0.04081632653061224, 0.06832298136645963, 0.06201550387596899, 0.04716981132075472, 0.056818181818181816, 0.013333333333333334, 0.028985507246376812, 0.03278688524590164, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0625, 0.03571428571428571, 0.0, 0.0, 0.043478260869565216, 0.0, 0.05555555555555555, 0.0, 0.0625, 0.07142857142857142, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#BORDER_DEATH_PROBABILITIES = [-99, 0.3290203327171904, 0.5868263473053892, 0.48148148148148145, 0.4375, 0.42424242424242425, 0.2222222222222222, 0.35714285714285715, 0.2222222222222222, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.05133928571428571, 0.006134969325153374, 0.03468208092485549, 0.025735294117647058, 0.037037037037037035, 0.02247191011235955, 0.04081632653061224, 0.05, 0.05, 0.036585365853658534, 0.013888888888888888, 0.030303030303030304, 0.03389830508474576, 0.0, 0.0, 0.0, 0.05128205128205128, 0.0, 0.06451612903225806, 0.037037037037037035, 0.0, 0.0, 0.045454545454545456, 0.0, 0.05555555555555555, 0.0, 0.0625, 0.07142857142857142, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#from learn_params.py
#BORDER_DEATH_PROBABILITIES = [-99, 0.3290203327171904, 0.5868263473053892, 0.48148148148148145, 0.4375, 0.42424242424242425]
#NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.05133928571428571, 0.006134969325153374, 0.03468208092485549, 0.025735294117647058, 0.037037037037037035]

#BORDER_DEATH_PROBABILITIES = [-99, 0.3116591928251121, 0.5483870967741935, 0.5833333333333334, 0.8571428571428571, 1.0]
#NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.001880843060242297, 0.026442307692307692, 0.04918032786885246, 0.06818181818181818, 0.008]

#BORDER_DEATH_PROBABILITIES = [-99, 0.3290203327171904, 0.5868263473053892, 0.48148148148148145, 0.4375, 0.42424242424242425]
#NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.05133928571428571, 0.006134969325153374, 0.03468208092485549, 0.025735294117647058, 0.037037037037037035]


#BORDER_DEATH_PROBABILITIES = [-99, 0.8, 0.5, 0.3, 0.4, 0.8]
#NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.07, 0.025, 0.03, 0.03, 0.006]

#BORDER_DEATH_PROBABILITIES = [-99, 0.9430523917995444, 0.6785714285714286, 0.4444444444444444, 0.5, 1.0]
#NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.08235294117647059, 0.02284263959390863, 0.04150943396226415, 0.041237113402061855, 0.00684931506849315]
#from learn_params
#CLUTTER_COUNT_PRIOR = [0.7860256569933989, 0.17523975588491716 - .001, 0.031635321957902605, 0.004857391954166148, 0.0016191306513887158, 0.0003736455349358575, 0.00024909702329057166, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0]
#from learn_params1, not counting 'ignored' ground truth
CLUTTER_COUNT_PRIOR = [0.5424333167268651, 0.3045211109727239, 0.11010088429443268, 0.0298916427948686, 0.008718395815170008, 0.003113712791132146, 0.0009963880931622867, 0.00012454851164528583, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06]

p_clutter_likelihood = 1.0/float(1242*375)
#p_birth_likelihood = 0.035
p_birth_likelihood = 1.0/float(1242*375)


#Kalman filter defaults
#Think about doing this in a more principled way!!!
#P_default = np.array([[57.54277774, 0, 			 0, 0],
# 					  [0,          10, 			 0, 0],
# 					  [0, 			0, 17.86392672, 0],
# 					  [0, 			0, 			 0, 3]])
P_default = np.array([[40.64558317, 0, 			 0, 0],
 					  [0,          10, 			 0, 0],
 					  [0, 			0, 5.56278505, 0],
 					  [0, 			0, 			 0, 3]])

#regionlet detection with score > 2.0:
#from learn_params
#R_default = np.array([[  5.60121574e+01,  -3.60666228e-02],
# 					  [ -3.60666228e-02,   1.64772050e+01]])
#from learn_params1, not counting 'ignored' ground truth
#R_default = np.array([[ 40.64558317,   0.14036472],
# 					  [  0.14036472,   5.56278505]])
R_default = np.array([[ 0.0,   0.0],
 					  [ 0.0,   0.0]])


#learned from all GT
#Q_default = np.array([[ 84.30812679,  84.21851631,  -4.01491901,  -8.5737873 ],
# 					  [ 84.21851631,  84.22312789,  -3.56066467,  -8.07744876],
# 					  [ -4.01491901,  -3.56066467,   4.59923143,   5.19622064],
# 					  [ -8.5737873 ,  -8.07744876,   5.19622064,   6.10733628]])
#also learned from all GT
Q_default = np.array([[  60.33442497,  102.95992102,   -5.50458177,   -0.22813535],
 					  [ 102.95992102,  179.84877761,  -13.37640528,   -9.70601621],
 					  [  -5.50458177,  -13.37640528,    4.56034398,    9.48945108],
 					  [  -0.22813535,   -9.70601621,    9.48945108,   22.32984314]])

Q_default = 4*Q_default

#measurement function matrix
H = np.array([[1.0,  0.0, 0.0, 0.0],
              [0.0,  0.0, 1.0, 0.0]])	

USE_LEARNED_DEATH_PROBABILITIES = True

#Gamma distribution parameters for calculating target death probabilities
alpha_death = 2.0
beta_death = 1.0
theta_death = 1.0/beta_death

print Q_default
print R_default

#for only displaying targets older than this
min_target_age = .2

#state parameters, during data generation uniformly sample new targets from range:
min_pos = -5.0
max_pos = 5.0
min_vel = -1.0
max_vel = 1.0

#The maximum allowed distance for a ground truth target and estimated target
#to be associated with each other when calculating MOTA and MOTP
MAX_ASSOCIATION_DIST = 1

CAMERA_PIXEL_WIDTH = 1242
CAMERA_PIXEL_HEIGHT = 375

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color




class Target:
	def __init__(self, cur_time, id_, measurement = None, width=-1, height=-1):
#		if measurement is None: #for data generation
#			position = np.random.uniform(min_pos,max_pos)
#			velocity = np.random.uniform(min_vel,max_vel)
#			self.x = np.array([[position], [velocity]])
#			self.P = P_default
#		else:
		assert(measurement != None)
		self.x = np.array([[measurement[0]], [0], [measurement[1]], [0]])
		self.P = P_default

		self.width = width
		self.height = height

		assert(self.x.shape == (4, 1))
		self.birth_time = cur_time
		#Time of the last measurement data association with this target
		self.last_measurement_association = cur_time
		self.id_ = id_ #named id_ to avoid clash with built in id
		self.death_prob = -1 #calculate at every time instance

		self.all_states = [(self.x, self.width, self.height)]
		self.all_time_stamps = [round(cur_time, 1)]

		self.measurements = []
		self.measurement_time_stamps = []

		#if target's predicted location is offscreen, set to True and then kill
		self.offscreen = False

		self.updated_this_time_instance = True

	def near_border(self):
		near_border = False
		x1 = self.x[0][0] - self.width/2.0
		x2 = self.x[0][0] + self.width/2.0
		y1 = self.x[2][0] - self.height/2.0
		y2 = self.x[2][0] + self.height/2.0
		if(x1 < 10 or x2 > (CAMERA_PIXEL_WIDTH - 15) or y1 < 10 or y2 > (CAMERA_PIXEL_HEIGHT - 15)):
			near_border = True
		return near_border


	def kf_update(self, measurement, width, height, cur_time, meas_noise_cov):
		""" Perform Kalman filter update step and replace predicted position for the current time step
		with the updated position in self.all_states
		Input:
		- measurement: the measurement (numpy array)
		- cur_time: time when the measurement was taken (float)
!!!!!!!!!PREDICTION HAS BEEN RUN AT THE BEGINNING OF TIME STEP FOR EVERY TARGET!!!!!!!!!
		"""
		reformat_meas = np.array([[measurement[0]],
								  [measurement[1]]])
		assert(self.x.shape == (4, 1))
		if USE_CONSTANT_R:
			S = np.dot(np.dot(H, self.P), H.T) + R_default
		else:
			S = np.dot(np.dot(H, self.P), H.T) + meas_noise_cov
		K = np.dot(np.dot(self.P, H.T), inv(S))
		residual = reformat_meas - np.dot(H, self.x)
		updated_x = self.x + np.dot(K, residual)
	#	updated_self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, H)), self.P) #NUMERICALLY UNSTABLE!!!!!!!!
		updated_P = self.P - np.dot(np.dot(K, S), K.T) #not sure if this is numerically stable!!
		self.x = updated_x
		self.P = updated_P
		self.width = width
		self.height = height
		assert(self.all_time_stamps[-1] == round(cur_time, 1) and self.all_time_stamps[-2] != round(cur_time, 1))
		assert(self.x.shape == (4, 1)), (self.x.shape, np.dot(K, residual).shape)

		self.all_states[-1] = (self.x, self.width, self.height)
		self.updated_this_time_instance = True
		self.last_measurement_association = cur_time

	def kf_predict(self, dt, cur_time):
		"""
		Run kalman filter prediction on this target
		Inputs:
			-dt: time step to run prediction on
			-cur_time: the time the prediction is made for
		"""
		assert(self.all_time_stamps[-1] == round((cur_time - dt), 1))
		F = np.array([[1.0,  dt, 0.0, 0.0],
		      		  [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0,  dt],
                      [0.0, 0.0, 0.0, 1.0]])
		x_predict = np.dot(F, self.x)
		P_predict = np.dot(np.dot(F, self.P), F.T) + Q_default
		self.x = x_predict
		self.P = P_predict
		self.all_states.append((self.x, self.width, self.height))
		self.all_time_stamps.append(round(cur_time, 1))

		if(self.x[0][0]<0 or self.x[0][0]>=CAMERA_PIXEL_WIDTH or \
		   self.x[2][0]<0 or self.x[2][0]>=CAMERA_PIXEL_HEIGHT):
#			print '!'*40, "TARGET IS OFFSCREEN", '!'*40
			self.offscreen = True

		assert(self.x.shape == (4, 1))
		self.updated_this_time_instance = False



###################	def target_death_prob(self, cur_time, prev_time):
###################		""" Calculate the target death probability if this was the only target.
###################		Actual target death probability will be (return_val/number_of_targets)
###################		because we limit ourselves to killing a max of one target per measurement.
###################
###################		Input:
###################		- cur_time: The current measurement time (float)
###################		- prev_time: The previous time step when a measurement was received (float)
###################
###################		Return:
###################		- death_prob: Probability of target death if this is the only target (float)
###################		"""
###################
###################		#scipy.special.gdtrc(b, a, x) calculates 
###################		#integral(gamma_dist(k = a, theta = b))from x to infinity
###################		last_assoc = self.last_measurement_association
###################
###################		#I think this is correct
###################		death_prob = gdtrc(theta_death, alpha_death, prev_time - last_assoc) \
###################				   - gdtrc(theta_death, alpha_death, cur_time - last_assoc)
###################		death_prob /= gdtrc(theta_death, alpha_death, prev_time - last_assoc)
###################		return death_prob
###################
####################		#this is used in paper's code
####################		time_step = cur_time - prev_time
####################	
####################		death_prob = gdtrc(theta_death, alpha_death, cur_time - last_assoc) \
####################				   - gdtrc(theta_death, alpha_death, cur_time - last_assoc + time_step)
####################		death_prob /= gdtrc(theta_death, alpha_death, cur_time - last_assoc)
####################		return death_prob
	def target_death_prob(self, cur_time, prev_time):
		""" Calculate the target death probability if this was the only target.
		Actual target death probability will be (return_val/number_of_targets)
		because we limit ourselves to killing a max of one target per measurement.

		Input:
		- cur_time: The current measurement time (float)
		- prev_time: The previous time step when a measurement was received (float)

		Return:
		- death_prob: Probability of target death if this is the only target (float)
		"""

##################		#scipy.special.gdtrc(b, a, x) calculates 
##################		#integral(gamma_dist(k = a, theta = b))from x to infinity
##################		last_assoc = self.last_measurement_association
##################
##################		#I think this is correct
##################		death_prob = gdtrc(theta_death, alpha_death, prev_time - last_assoc) \
##################				   - gdtrc(theta_death, alpha_death, cur_time - last_assoc)
##################		death_prob /= gdtrc(theta_death, alpha_death, prev_time - last_assoc)
##################		return death_prob
		if(self.offscreen == True):
			cur_death_prob = 1.0
		else:
			frames_since_last_assoc = int(round((cur_time - self.last_measurement_association)/default_time_step))
			assert(abs(float(frames_since_last_assoc) - (cur_time - self.last_measurement_association)/default_time_step) < .00000001)
			if(self.near_border()):
				if frames_since_last_assoc < len(BORDER_DEATH_PROBABILITIES):
					cur_death_prob = BORDER_DEATH_PROBABILITIES[frames_since_last_assoc]
				else:
					cur_death_prob = BORDER_DEATH_PROBABILITIES[-1]
#					cur_death_prob = 1.0
			else:
				if frames_since_last_assoc < len(NOT_BORDER_DEATH_PROBABILITIES):
					cur_death_prob = NOT_BORDER_DEATH_PROBABILITIES[frames_since_last_assoc]
				else:
					cur_death_prob = NOT_BORDER_DEATH_PROBABILITIES[-1]
#					cur_death_prob = 1.0

		assert(cur_death_prob >= 0.0 and cur_death_prob <= 1.0), cur_death_prob
		return cur_death_prob

class Measurement:
    #a collection of measurements at a single time instance
    def __init__(self, time = -1):
        #self.val is a list of numpy arrays of measurement x, y locations
        self.val = []
        #list of widths of each bounding box
        self.widths = []
        #list of widths of each bounding box        
        self.heights = []
        #list of scores for each individual measurement
        self.scores = []
        self.time = time

class TargetSet:
	"""
	Contains ground truth states for all targets.  Also contains all generated measurements.
	"""

	def __init__(self):
		self.living_targets = []
		self.all_targets = [] #alive and dead targets

		self.living_count = 0 #number of living targets
		self.total_count = 0 #number of living targets plus number of dead targets
		self.measurements = [] #generated measurements for a generative TargetSet 

		self.parent_target_set = None 

		self.living_targets_q = deque([-1 for i in range(ONLINE_DELAY)])

	def create_child(self):
		child_target_set = TargetSet()
		child_target_set.parent_target_set = self
		child_target_set.total_count = self.total_count
		child_target_set.living_count = self.living_count
		child_target_set.all_targets = copy.deepcopy(self.living_targets)
		for target in child_target_set.all_targets:
			child_target_set.living_targets.append(target)
		child_target_set.living_targets_q = copy.deepcopy(self.living_targets_q)
		return child_target_set

	def create_new_target(self, measurement, width, height, cur_time):
		if RUN_ONLINE:
			global NEXT_TARGET_ID
			new_target = Target(cur_time, NEXT_TARGET_ID, np.squeeze(measurement), width, height)
			NEXT_TARGET_ID += 1
		else:
			new_target = Target(cur_time, self.total_count, np.squeeze(measurement), width, height)
		self.living_targets.append(new_target)
		self.all_targets.append(new_target)
		self.living_count += 1
		self.total_count += 1
		if not USE_CREATE_CHILD:
			assert(len(self.living_targets) == self.living_count and len(self.all_targets) == self.total_count)


	def kill_target(self, living_target_index):
		"""
		Kill target self.living_targets[living_target_index], note that living_target_index
		may not be the target's id_ (or index in all_targets)
		"""

		#kf predict was run for this time instance, but the target actually died, so remove the predicted state
		del self.living_targets[living_target_index].all_states[-1]
		del self.living_targets[living_target_index].all_time_stamps[-1]

		del self.living_targets[living_target_index]

		self.living_count -= 1
		if not USE_CREATE_CHILD:
			assert(len(self.living_targets) == self.living_count and len(self.all_targets) == self.total_count)

	def plot_all_target_locations(self, title):
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		for i in range(self.total_count):
			life = len(self.all_targets[i].all_states) #length of current targets life 
			locations_1D =  [self.all_targets[i].all_states[j][0] for j in range(life)]
			ax.plot(self.all_targets[i].all_time_stamps, locations_1D,
					'-o', label='Target %d' % i)

		legend = ax.legend(loc='lower left', shadow=True)
		plt.title('%s, unique targets = %d, #targets alive = %d' % \
			(title, self.total_count, self.living_count)) # subplot 211 title

	def plot_generated_measurements(self):
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		time_stamps = [self.measurements[i].time for i in range(len(self.measurements))
												 for j in range(len(self.measurements[i].val))]
		locations = [self.measurements[i].val[j][0] for i in range(len(self.measurements))
													for j in range(len(self.measurements[i].val))]
		ax.plot(time_stamps, locations,'o')
		plt.title('Generated Measurements') 

	def collect_ancestral_targets(self, descendant_target_ids=[]):
		"""
		Inputs:
		- descendant_target_ids: a list of target ids that exist in the calling child's all_targets list
			(or the all_targets list of a descendant of the calling child)

		Outputs:
		- every_target: every target in this TargetSet's all_targets list and
		#every target in any of this TargetSet's ancestors' all_targets lists that does not
		#appear in the all_targets list of a descendant
		"""
		every_target = []
		found_target_ids = descendant_target_ids
		for target in self.all_targets:
			if(not target.id_ in found_target_ids):
				every_target.append(target)
				found_target_ids.append(target.id_)
		if self.parent_target_set == None:
			return every_target
		else:
			ancestral_targets = self.parent_target_set.collect_ancestral_targets(found_target_ids)

		every_target = every_target + ancestral_targets # + operator used to concatenate lists!
		return every_target


	def write_online_results(self, online_results_filename, frame_idx, total_frame_count):
		if frame_idx == ONLINE_DELAY:
			f = open(online_results_filename, "w") #write over old results if first frame
		else:
			f = open(online_results_filename, "a") #write at end of file

		if ONLINE_DELAY == 0:
			for target in self.living_targets:
				assert(target.all_time_stamps[-1] == round(frame_idx*default_time_step, 1))
				x_pos = target.all_states[-1][0][0][0]
				y_pos = target.all_states[-1][0][2][0]
				width = target.all_states[-1][1]
				height = target.all_states[-1][2]

				left = x_pos - width/2.0
				top = y_pos - height/2.0
				right = x_pos + width/2.0
				bottom = y_pos + height/2.0		 
				f.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
					(frame_idx, target.id_, left, top, right, bottom))

		else:
			print self.living_targets_q
			(delayed_frame_idx, delayed_liv_targets) = self.living_targets_q[0]
			print delayed_frame_idx
			print delayed_liv_targets
			assert(delayed_frame_idx == frame_idx - ONLINE_DELAY), (delayed_frame_idx, frame_idx, ONLINE_DELAY)
			for target in delayed_liv_targets:
				assert(target.all_time_stamps[-1] == round((frame_idx - ONLINE_DELAY)*default_time_step, 1)), (target.all_time_stamps[-1], frame_idx, ONLINE_DELAY, round((frame_idx - ONLINE_DELAY)*default_time_step, 1))
				x_pos = target.all_states[-1][0][0][0]
				y_pos = target.all_states[-1][0][2][0]
				width = target.all_states[-1][1]
				height = target.all_states[-1][2]

				left = x_pos - width/2.0
				top = y_pos - height/2.0
				right = x_pos + width/2.0
				bottom = y_pos + height/2.0		 
				f.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
					(frame_idx - ONLINE_DELAY, target.id_, left, top, right, bottom))

			if frame_idx == total_frame_count - 1:
				q_idx = 1
				for cur_frame_idx in range(frame_idx - ONLINE_DELAY + 1, total_frame_count - 1):
					print '-'*20
					print cur_frame_idx
					print frame_idx - ONLINE_DELAY + 1
					print total_frame_count
					print q_idx
					print len(self.living_targets_q)
					(delayed_frame_idx, delayed_liv_targets) = self.living_targets_q[q_idx]
					q_idx+=1
					assert(delayed_frame_idx == cur_frame_idx), (delayed_frame_idx, cur_frame_idx, ONLINE_DELAY)
					for target in delayed_liv_targets:
						assert(target.all_time_stamps[-1] == round((cur_frame_idx)*default_time_step, 1))
						x_pos = target.all_states[-1][0][0][0]
						y_pos = target.all_states[-1][0][2][0]
						width = target.all_states[-1][1]
						height = target.all_states[-1][2]

						left = x_pos - width/2.0
						top = y_pos - height/2.0
						right = x_pos + width/2.0
						bottom = y_pos + height/2.0		 
						f.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
							(cur_frame_idx, target.id_, left, top, right, bottom))
				for target in self.living_targets:
					assert(target.all_time_stamps[-1] == round(frame_idx*default_time_step, 1))
					x_pos = target.all_states[-1][0][0][0]
					y_pos = target.all_states[-1][0][2][0]
					width = target.all_states[-1][1]
					height = target.all_states[-1][2]

					left = x_pos - width/2.0
					top = y_pos - height/2.0
					right = x_pos + width/2.0
					bottom = y_pos + height/2.0		 
					f.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
						(frame_idx, target.id_, left, top, right, bottom))

	def write_targets_to_KITTI_format(self, num_frames, filename):
		if USE_CREATE_CHILD:
			every_target = self.collect_ancestral_targets()
			f = open(filename, "w")
			for frame_idx in range(num_frames):
				timestamp = round(frame_idx*default_time_step, 1)
				for target in every_target:
					if timestamp in target.all_time_stamps:
						x_pos = target.all_states[target.all_time_stamps.index(timestamp)][0][0][0]
						y_pos = target.all_states[target.all_time_stamps.index(timestamp)][0][2][0]
						width = target.all_states[target.all_time_stamps.index(timestamp)][1]
						height = target.all_states[target.all_time_stamps.index(timestamp)][2]

						left = x_pos - width/2.0
						top = y_pos - height/2.0
						right = x_pos + width/2.0
						bottom = y_pos + height/2.0		 
						f.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
							(frame_idx, target.id_, left, top, right, bottom))
	#					left = target.x[0][0] - target.width/2
	#					top = target.x[2][0] - target.height/2
	#					right = target.x[0][0] + target.width/2
	#					bottom = target.x[2][0] + target.height/2		 
	#					f.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
	#						(frame_idx, target.id_, left, top, right, bottom))
			f.close()

		else:
			f = open(filename, "w")
			for frame_idx in range(num_frames):
				timestamp = round(frame_idx*default_time_step, 1)
				for target in self.all_targets:
					if timestamp in target.all_time_stamps:
						x_pos = target.all_states[target.all_time_stamps.index(timestamp)][0][0][0]
						y_pos = target.all_states[target.all_time_stamps.index(timestamp)][0][2][0]
						width = target.all_states[target.all_time_stamps.index(timestamp)][1]
						height = target.all_states[target.all_time_stamps.index(timestamp)][2]

						left = x_pos - width/2.0
						top = y_pos - height/2.0
						right = x_pos + width/2.0
						bottom = y_pos + height/2.0		 
						f.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
							(frame_idx, target.id_, left, top, right, bottom))
	#					left = target.x[0][0] - target.width/2
	#					top = target.x[2][0] - target.height/2
	#					right = target.x[0][0] + target.width/2
	#					bottom = target.x[2][0] + target.height/2		 
	#					f.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
	#						(frame_idx, target.id_, left, top, right, bottom))
			f.close()


class Particle:
	def __init__(self, id_):
		#Targets tracked by this particle
		self.targets = TargetSet()

		self.importance_weight = 1.0/N_PARTICLES
		self.likelihood_DOUBLE_CHECK_ME = -1
		#cache for memoizing association likelihood computation
		self.assoc_likelihood_cache = {}

		self.id_ = id_ #will be the same as the parent's id when copying in create_child

		self.parent_id = -1

		#for debugging
		self.c_debug = -1
		self.imprt_re_weight_debug = -1
		self.pi_birth_debug = -1
		self.pi_clutter_debug = -1
		self.pi_targets_debug = []

	def create_child(self):
		global NEXT_PARTICLE_ID
		child_particle = Particle(NEXT_PARTICLE_ID)
		NEXT_PARTICLE_ID += 1
		child_particle.importance_weight = self.importance_weight
		child_particle.targets = self.targets.create_child()
		return child_particle

	def create_new_target(self, measurement, width, height, cur_time):
		self.targets.create_new_target(measurement, width, height, cur_time)

	def update_target_death_probabilities(self, cur_time, prev_time):
		for target in self.targets.living_targets:
			target.death_prob = target.target_death_prob(cur_time, prev_time)

	def sample_target_deaths(self):
		"""

		Implemented to possibly kill multiple targets at once, seems
		reasonbale but CHECK TECHNICAL DETAILS!!

		death_prob for every target should have already been calculated!!

		Input:
		- cur_time: The current measurement time (float)
		- prev_time: The previous time step when a measurement was received (float)

		"""
		original_num_targets = self.targets.living_count
		num_targets_killed = 0
		indices_to_kill = []
		for (index, cur_target) in enumerate(self.targets.living_targets):
			death_prob = cur_target.death_prob
			assert(death_prob < 1.0 and death_prob > 0.0)
			if (random.random() < death_prob):
				indices_to_kill.append(index)
				num_targets_killed += 1

		#important to delete largest index first to preserve values of the remaining indices
		for index in reversed(indices_to_kill):
			self.targets.kill_target(index)

		assert(self.targets.living_count == (original_num_targets - num_targets_killed))
		#print "targets killed = ", num_targets_killed




	def sample_data_assoc_and_death_mult_meas_per_time_proposal_distr_1(self, measurement_lists, \
		cur_time, measurement_scores):
		"""
		Input:
		- measurement_lists: a list where measurement_lists[i] is a list of all measurements from the current
			time instance from the ith measurement source (i.e. different object detection algorithms
			or different sensors)
		- measurement_scores: a list where measurement_scores[i] is a list containing scores for every measurement in
			measurement_list[i]

		Output:
		- measurement_associations: A list where measurement_associations[i] is a list of association values
			for each measurements in measurement_lists[i].  Association values correspond to:
			measurement_associations[i][j] = -1 -> measurement is clutter
			measurement_associations[i][j] = self.targets.living_count -> measurement is a new target
			measurement_associations[i][j] in range [0, self.targets.living_count-1] -> measurement is of
				particle.targets.living_targets[measurement_associations[i][j]]

		- imprt_re_weight: After processing this measurement the particle's
			importance weight will be:
			new_importance_weight = old_importance_weight * imprt_re_weight
		- targets_to_kill: a list containing the indices of targets that should be killed, beginning
			with the smallest index in increasing order, e.g. [0, 4, 6, 33]
		"""

		#get death probabilities for each target in a numpy array
		num_targs = self.targets.living_count
		p_target_deaths = []
		for target in self.targets.living_targets:
			p_target_deaths.append(target.death_prob)
			assert(p_target_deaths[len(p_target_deaths) - 1] >= 0 and p_target_deaths[len(p_target_deaths) - 1] <= 1)

		if InfLoopDEBUG:
			with open(debugInfLoopFile, "a") as myfile:
				myfile.write("location 4\n")

		(targets_to_kill, measurement_associations, proposal_probability, unassociated_target_death_probs) = \
			self.sample_proposal_distr3(measurement_lists, self.targets.living_count, p_target_deaths, \
										cur_time, measurement_scores)


		if InfLoopDEBUG:
			with open(debugInfLoopFile, "a") as myfile:
				myfile.write("location 5\n")

		living_target_indices = []
		for i in range(self.targets.living_count):
			if(not i in targets_to_kill):
				living_target_indices.append(i)

#		exact_probability = self.get_exact_prob_hidden_and_data(measurement_list, living_target_indices, self.targets.living_count, 
#												 measurement_associations, p_target_deaths)
		exact_probability = 1.0
		for meas_source_index in range(len(measurement_lists)):
			cur_assoc_prob = self.get_exact_prob_hidden_and_data(meas_source_index, measurement_lists[meas_source_index], \
				living_target_indices, self.targets.living_count, measurement_associations[meas_source_index],\
				unassociated_target_death_probs, measurement_scores[meas_source_index], SCORE_INTERVALS[meas_source_index])
			exact_probability *= cur_assoc_prob


		if InfLoopDEBUG:
			with open(debugInfLoopFile, "a") as myfile:
				myfile.write("location 6\n")

		exact_death_prob = self.calc_death_prior(living_target_indices, p_target_deaths)
		exact_probability *= exact_death_prob

		assert(num_targs == self.targets.living_count)
		#double check targets_to_kill is sorted
		assert(all([targets_to_kill[i] <= targets_to_kill[i+1] for i in xrange(len(targets_to_kill)-1)]))

		imprt_re_weight = exact_probability/proposal_probability

		assert(imprt_re_weight != 0.0), (exact_probability, proposal_probability)

		self.likelihood_DOUBLE_CHECK_ME = exact_probability

		return (measurement_associations, targets_to_kill, imprt_re_weight)


	def associate_measurements_proposal_distr3(self, meas_source_index, measurement_list, total_target_count, \
		p_target_deaths, measurement_scores):

		"""
		Try sampling associations with each measurement sequentially
		Input:
		- measurement_list: a list of all measurements from the current time instance
		- total_target_count: the number of living targets on the previous time instace
		- p_target_deaths: a list of length len(total_target_count) where 
			p_target_deaths[i] = the probability that target i has died between the last
			time instance and the current time instance

		Output:
		- list_of_measurement_associations: list of associations for each measurement
		- proposal_probability: proposal probability of the sampled deaths and associations
			
		"""
		list_of_measurement_associations = []
		proposal_probability = 1.0

		#sample measurement associations
		birth_count = 0
		clutter_count = 0
		remaining_meas_count = len(measurement_list)
		if InfLoopDEBUG or InfLoopDEBUG1 or InfLoopDEBUG3:
			with open(debugInfLoopFile, "a") as myfile:
				myfile.write("number of measurements = %d and number of targets = %d\n" % (len(measurement_list), total_target_count))

		for (index, cur_meas) in enumerate(measurement_list):
			if InfLoopDEBUG or InfLoopDEBUG1 or InfLoopDEBUG2 or InfLoopDEBUG3:
				with open(debugInfLoopFile, "a") as myfile:
					myfile.write("processing measurement %d\n" % index)

			score_index = get_score_index(SCORE_INTERVALS[meas_source_index], measurement_scores[index])
			#create proposal distribution for the current measurement
			#compute target association proposal probabilities
			proposal_distribution_list = []
			for target_index in range(total_target_count):
				if InfLoopDEBUG2 and (index == 0):
					with open(debugInfLoopFile, "a") as myfile:
						myfile.write("about to add target %d to proposal distribution\n" % target_index)

				cur_target_likelihood = self.memoized_assoc_likelihood(cur_meas, meas_source_index, target_index, MEAS_NOISE_COVS[meas_source_index][score_index], score_index)
				
				if InfLoopDEBUG2 and (index == 0):
					with open(debugInfLoopFile, "a") as myfile:
						myfile.write("got target %d's likelihood\n" % target_index)


				targ_likelihoods_summed_over_meas = 0.0
				for meas_index in range(len(measurement_list)):
					temp_score_index = get_score_index(SCORE_INTERVALS[meas_source_index], measurement_scores[meas_index]) #score_index for the meas_index in this loop
					if InfLoopDEBUG2 and (index == 0):
						with open(debugInfLoopFile, "a") as myfile:
							myfile.write("got mesaurement %d's score index\n" % meas_index)
				
					targ_likelihoods_summed_over_meas += self.memoized_assoc_likelihood(measurement_list[meas_index], meas_source_index, target_index,  MEAS_NOISE_COVS[meas_source_index][temp_score_index], temp_score_index)

					if InfLoopDEBUG2 and (index == 0):
						with open(debugInfLoopFile, "a") as myfile:
							myfile.write("got likelihood mesaurement %d\n" % meas_index)

				if InfLoopDEBUG2 and (index == 0):
					with open(debugInfLoopFile, "a") as myfile:
						myfile.write("targ_likelihoods_summed_over_meas = %f, target_index = %d, p_target_deaths[target_index] = %f, (target_index in list_of_measurement_associations) = %r\n" % (targ_likelihoods_summed_over_meas, target_index, p_target_deaths[target_index], (target_index in list_of_measurement_associations)))

				if((targ_likelihoods_summed_over_meas != 0.0) and (not target_index in list_of_measurement_associations)\
					and p_target_deaths[target_index] < 1.0):
					cur_target_prior = TARGET_EMISSION_PROBS[meas_source_index][score_index]*cur_target_likelihood \
									  /targ_likelihoods_summed_over_meas
					if InfLoopDEBUG2 and (index == 0):
						with open(debugInfLoopFile, "a") as myfile:
							myfile.write("target_prior set in if statement")

#					cur_target_prior = P_TARGET_EMISSION*cur_target_likelihood \
#									  /targ_likelihoods_summed_over_meas
				else:
					cur_target_prior = 0.0
					if InfLoopDEBUG2 and (index == 0):
						with open(debugInfLoopFile, "a") as myfile:
							myfile.write("target_prior set in else statement")


				proposal_distribution_list.append(cur_target_likelihood*cur_target_prior)
				if InfLoopDEBUG1 or InfLoopDEBUG2:
					with open(debugInfLoopFile, "a") as myfile:
						myfile.write("added target %d to proposal distribution\n" % target_index)

			#compute birth association proposal probability
			cur_birth_prior = 0.0
			for i in range(birth_count+1, min(len(BIRTH_PROBABILITIES[meas_source_index][score_index]), remaining_meas_count + birth_count + 1)):
				cur_birth_prior += BIRTH_PROBABILITIES[meas_source_index][score_index][i]*(i - birth_count)/remaining_meas_count 
			proposal_distribution_list.append(cur_birth_prior*p_birth_likelihood)

			if InfLoopDEBUG1:
				with open(debugInfLoopFile, "a") as myfile:
					myfile.write("added birth probability to proposal distribution\n")


			#compute clutter association proposal probability
			cur_clutter_prior = 0.0
			for i in range(clutter_count+1, min(len(CLUTTER_PROBABILITIES[meas_source_index][score_index]), remaining_meas_count + clutter_count + 1)):
				cur_clutter_prior += CLUTTER_PROBABILITIES[meas_source_index][score_index][i]*(i - clutter_count)/remaining_meas_count 
			proposal_distribution_list.append(cur_clutter_prior*p_clutter_likelihood)

			if InfLoopDEBUG1:
				with open(debugInfLoopFile, "a") as myfile:
					myfile.write("added clutter probability to proposal distribution\n")


			#normalize the proposal distribution
			proposal_distribution = np.asarray(proposal_distribution_list)
			assert(np.sum(proposal_distribution) != 0.0), (len(proposal_distribution), proposal_distribution, birth_count, clutter_count, len(measurement_list), total_target_count)

			if InfLoopDEBUG1:
				with open(debugInfLoopFile, "a") as myfile:
					myfile.write("converted proposal distribution to np array\n")


			proposal_distribution /= float(np.sum(proposal_distribution))
			assert(len(proposal_distribution) == total_target_count+2)

			if InfLoopDEBUG1:
				with open(debugInfLoopFile, "a") as myfile:
					myfile.write("normalized proposal distribution\n")

			sampled_assoc_idx = np.random.choice(len(proposal_distribution),
													p=proposal_distribution)

			if InfLoopDEBUG1 or InfLoopDEBUG3:
				with open(debugInfLoopFile, "a") as myfile:
					myfile.write("sampled index %d from proposal distribution\n" % sampled_assoc_idx)

			if(sampled_assoc_idx <= total_target_count): #target or birth association
				list_of_measurement_associations.append(sampled_assoc_idx)
				if(sampled_assoc_idx == total_target_count):
					birth_count += 1
			else: #clutter association
				assert(sampled_assoc_idx == total_target_count+1)
				list_of_measurement_associations.append(-1)
				clutter_count += 1
			proposal_probability *= proposal_distribution[sampled_assoc_idx]

			remaining_meas_count -= 1
			if InfLoopDEBUG1 or InfLoopDEBUG3:
				with open(debugInfLoopFile, "a") as myfile:
					myfile.write("finished processing measurement %d\n" % index)

		assert(remaining_meas_count == 0)
		return(list_of_measurement_associations, proposal_probability)

	def sample_proposal_distr3(self, measurement_lists, total_target_count, 
							   p_target_deaths, cur_time, measurement_scores):
		"""
		Try sampling associations with each measurement sequentially
		Input:
		- measurement_lists: type list, measurement_lists[i] is a list of all measurements from the current
			time instance from the ith measurement source (i.e. different object detection algorithms
			or different sensors)
		- measurement_scores: type list, measurement_scores[i] is a list containing scores for every measurement in
			measurement_list[i]
		- total_target_count: the number of living targets on the previous time instace
		- p_target_deaths: a list of length len(total_target_count) where 
			p_target_deaths[i] = the probability that target i has died between the last
			time instance and the current time instance

		Output:
		- targets_to_kill: a list of targets that have been sampled to die (not killed yet)
		- measurement_associations: type list, measurement_associations[i] is a list of associations for  
			the measurements in measurement_lists[i]
		- proposal_probability: proposal probability of the sampled deaths and associations
			
		"""
		assert(len(measurement_lists) == len(measurement_scores))
		measurement_associations = []
		proposal_probability = 1.0
		for meas_source_index in range(len(measurement_lists)):
			(cur_associations, cur_proposal_prob) = self.associate_measurements_proposal_distr3\
				(meas_source_index, measurement_lists[meas_source_index], total_target_count, \
				 p_target_deaths, measurement_scores[meas_source_index])
			measurement_associations.append(cur_associations)
			proposal_probability *= cur_proposal_prob

		assert(len(measurement_associations) == len(measurement_lists))
		if InfLoopDEBUG:
			with open(debugInfLoopFile, "a") as myfile:
				myfile.write("location 7\n")

############################################################################################################
		#sample target deaths from unassociated targets
		unassociated_targets = []
		unassociated_target_death_probs = []

		for i in range(total_target_count):
			target_unassociated = True
			for meas_source_index in range(len(measurement_associations)):
				if (i in measurement_associations[meas_source_index]):
					target_unassociated = False
			if target_unassociated:
				unassociated_targets.append(i)
				unassociated_target_death_probs.append(p_target_deaths[i])
			else:
				unassociated_target_death_probs.append(0.0)

		if InfLoopDEBUG:
			with open(debugInfLoopFile, "a") as myfile:
				myfile.write("location 8\n")


		if USE_LEARNED_DEATH_PROBABILITIES:
			(targets_to_kill, death_probability) =  \
				self.sample_target_deaths_proposal3(unassociated_targets, cur_time)
		else:
			(targets_to_kill, death_probability) =  \
				self.sample_target_deaths_proposal2(unassociated_targets, cur_time)

		#probability of sampling all associations
		proposal_probability *= death_probability
		assert(proposal_probability != 0.0)

		#debug
		for meas_source_index in range(len(measurement_associations)):
			for i in range(total_target_count):
				assert(measurement_associations[meas_source_index].count(i) == 0 or \
					   measurement_associations[meas_source_index].count(i) == 1), (measurement_associations[meas_source_index],  measurement_list, total_target_count, p_target_deaths)
		#done debug

		return (targets_to_kill, measurement_associations, proposal_probability, unassociated_target_death_probs)


	def sample_target_deaths_proposal3(self, unassociated_targets, cur_time):
		"""
		Sample target deaths, given they have not been associated with a measurement, using probabilities
		learned from data.
		Also kill all targets that are offscreen.

		Inputs:
		- unassociated_targets: a list of target indices that have not been associated with a measurement

		Output:
		- targets_to_kill: a list of targets that have been sampled to die (not killed yet)
		- probability_of_deaths: the probability of the sampled deaths
		"""
		targets_to_kill = []
		probability_of_deaths = 1.0

		for target_idx in range(len(self.targets.living_targets)):
			#kill offscreen targets with probability 1.0
			if(self.targets.living_targets[target_idx].offscreen == True):
				targets_to_kill.append(target_idx)
			elif(target_idx in unassociated_targets):
				cur_death_prob = self.targets.living_targets[target_idx].death_prob
				if(random.random() < cur_death_prob):
					targets_to_kill.append(target_idx)
					probability_of_deaths *= cur_death_prob
				else:
					probability_of_deaths *= (1 - cur_death_prob)
		return (targets_to_kill, probability_of_deaths)

	def calc_death_prior(self, living_target_indices, p_target_deaths):
		death_prior = 1.0
		for (cur_target_index, cur_target_death_prob) in enumerate(p_target_deaths):
			if cur_target_index in living_target_indices:
				death_prior *= (1.0 - cur_target_death_prob)
				assert((1.0 - cur_target_death_prob) != 0.0), cur_target_death_prob
			else:
				death_prior *= cur_target_death_prob
				assert((cur_target_death_prob) != 0.0), cur_target_death_prob

		return death_prior

	def get_prior(self, living_target_indices, total_target_count, number_measurements, 
				 measurement_associations, p_target_deaths, target_emission_probs, 
				 birth_count_priors, clutter_count_priors, measurement_scores, score_intervals):
		"""
DON"T THINK THIS BELONGS IN PARTICLE, OR PARAMETERS COULD BE CLEANED UP
		REDOCUMENT

		Input: 
		- living_target_indices: a list of indices of targets from last time instance that are still alive
		- total_target_count: the number of living targets on the previous time instace
		- number_measurements: the number of measurements on this time instance
		- measurement_associations: a list of association values for each measurement. Each association has the value
			of a living target index (index from last time instance), target birth (total_target_count), 
			or clutter (-1)
		-p_target_deaths: a list of length len(number_targets) where 
			p_target_deaths[i] = the probability that target i has died between the last
			time instance and the current time instance
		-p_target_emission: the probability that a target will emit a measurement on a 
			time instance (the same for all targets and time instances)
		-birth_count_prior: a probability distribution, specified as a list, such that
			birth_count_prior[i] = the probability of i births during any time instance
		-clutter_count_prior: a probability distribution, specified as a list, such that
			clutter_count_prior[i] = the probability of i clutter measurements during 
			any time instance
		"""

		def nCr(n,r):
		    return math.factorial(n) / math.factorial(r) / math.factorial(n-r)

		def count_meas_orderings(M, T, b, c):
			"""
			We define target observation priors in terms of whether each target was observed and it
			is irrelevant which measurement the target is associated with.  Likewise, birth count priors
			and clutter count priors are defined in terms of total counts, not which specific measurements
			are associated with clutter and births.  This function counts the number of possible 
			measurement-association assignments given we have already chosen which targets are observed, 
			how many births occur, and how many clutter measurements are present.  The prior probability of
			observing T specific targets, b births, and c clutter observations given M measurements should
			be divided by the returned value to split the prior probability between possibilities.

			[
			*OLD EXPLANATION BELOW*:
			We view the the ordering of measurements on any time instance as arbitrary.  This
			function counts the number of possible measurement orderings given we have already
			chosen which targets are observed, how many births occur, and how many clutter 
			measurements are present.
			]
			
			Inputs:
			- M: the number of measurements
			- T: the number of observed targets
			- b: the number of birth associations
			- c: the number of clutter associations

			This must be true: M = T+b+c

			Output:
			- combinations: the number of measurement orderings as a float. The value is:
				combinations = nCr(M, T)*math.factorial(T)*nCr(M-T, b)

			"""
			assert(M == T + b + c)
			combinations = nCr(M, T)*math.factorial(T)*nCr(M-T, b)
			return float(combinations)


		assert(len(measurement_associations) == number_measurements)
		#numnber of targets from the last time instance that are still alive
		living_target_count = len(living_target_indices)
		#numnber of targets from the last time instance that died
		dead_target_count = total_target_count - living_target_count

		#count the number of unique target associations
		unique_assoc = set(measurement_associations)
		if(total_target_count in unique_assoc):
			unique_assoc.remove(total_target_count)
		if((-1) in unique_assoc):
			unique_assoc.remove((-1))

		#the number of targets we observed on this time instance
		observed_target_count = len(unique_assoc)

		#the number of target measurements by measurement score
		meas_counts_by_score = [0 for i in range(len(score_intervals))]
		for i in range(len(measurement_associations)):
			if measurement_associations[i] != -1 and measurement_associations[i] != total_target_count:
				index = get_score_index(score_intervals, measurement_scores[i])
				meas_counts_by_score[index] += 1

		#the number of targets we don't observe on this time instance
		#but are still alive on this time instance
		unobserved_target_count = living_target_count - observed_target_count
		#the number of new targets born on this time instance
		birth_count = measurement_associations.count(total_target_count)
		birth_counts_by_score = [0 for i in range(len(score_intervals))]
		for i in range(len(measurement_associations)):
			if measurement_associations[i] == total_target_count:
				index = get_score_index(score_intervals, measurement_scores[i])
				birth_counts_by_score[index] += 1
		#the number of clutter measurements on this time instance
		clutter_count = measurement_associations.count(-1)
		clutter_counts_by_score = [0 for i in range(len(score_intervals))]
		for i in range(len(measurement_associations)):
			if measurement_associations[i] == -1:
				index = get_score_index(score_intervals, measurement_scores[i])
				clutter_counts_by_score[index] += 1

		assert(observed_target_count + birth_count + clutter_count == number_measurements),\
			(number_measurements, observed_target_count, birth_count, clutter_count, \
			total_target_count, measurement_associations)

#		assert(len(p_target_deaths) == total_target_count)
		death_prior = self.calc_death_prior(living_target_indices, p_target_deaths)

		#the prior probability of this number of measurements with these associations
		#given these target deaths
		for i in range(len(score_intervals)):

			assert(0 <= clutter_counts_by_score[i] and clutter_counts_by_score[i] < len(clutter_count_priors[i])), clutter_counts_by_score[i]
			assert(0 <= birth_counts_by_score[i] and birth_counts_by_score[i] < len(birth_count_priors[i])), birth_counts_by_score[i]

		p_target_does_not_emit = 1.0 - sum(target_emission_probs)
		assoc_prior = (p_target_does_not_emit)**(unobserved_target_count) \
					  /count_meas_orderings(number_measurements, observed_target_count, \
						  					birth_count, clutter_count)
		for i in range(len(score_intervals)):
			assoc_prior *= target_emission_probs[i]**(meas_counts_by_score[i]) \
							  *birth_count_priors[i][birth_counts_by_score[i]] \
							  *clutter_count_priors[i][clutter_counts_by_score[i]] \
						  

		total_prior = death_prior * assoc_prior

		if total_prior == 0:
			for i in range(len(score_intervals)):
				print "for score interval beginning at", score_intervals[i]
				print "target emmission prob =", target_emission_probs[i]**(meas_counts_by_score[i])
				print "birth prior=", birth_count_priors[i][birth_counts_by_score[i]] 
				print "clutter prior=", clutter_count_priors[i][clutter_counts_by_score[i]] 

		assert(total_prior != 0.0), (death_prior, assoc_prior, target_emission_probs, birth_count_priors, clutter_count_priors)
#		return total_prior
		return assoc_prior

	def get_exact_prob_hidden_and_data(self, meas_source_index, measurement_list, living_target_indices, total_target_count,
									   measurement_associations, p_target_deaths, measurement_scores, score_intervals):
		"""
		REDOCUMENT, BELOW INCORRECT, not including death probability now
		Calculate p(data, associations, #measurements, deaths) as:
		p(data|deaths, associations, #measurements)*p(deaths)*p(associations, #measurements|deaths)
		Input:
		- measurement_list: a list of all measurements from the current time instance, from the measurement
			source with index meas_source_index
		- living_target_indices: a list of indices of targets from last time instance that are still alive
		- total_target_count: the number of living targets on the previous time instace
		- measurement_associations: a list of association values for each measurement. Each association has the value
			of a living target index (index from last time instance), target birth (total_target_count), 
			or clutter (-1)
		- p_target_deaths: a list of length len(total_target_count) where 
			p_target_deaths[i] = the probability that target i has died between the last
			time instance and the current time instance

		Return:
		- p(data, associations, #measurements, deaths)


		*note* p(data|deaths, associations, #measurements) is referred to as the likelihood and
		p(deaths)*p(associations, #measurements|deaths) as the prior, even though the number of measurements
		is part of the data (or an observed variable)
		"""

		prior = self.get_prior(living_target_indices, total_target_count, len(measurement_list), 
				 				   measurement_associations, p_target_deaths, TARGET_EMISSION_PROBS[meas_source_index], 
								   BIRTH_PROBABILITIES[meas_source_index], CLUTTER_PROBABILITIES[meas_source_index], measurement_scores, score_intervals)

#		hidden_state = HiddenState(living_target_indices, total_target_count, len(measurement_list), 
#				 				   measurement_associations, p_target_deaths, P_TARGET_EMISSION, 
#								   BIRTH_COUNT_PRIOR, CLUTTER_COUNT_PRIOR)
#		priorA = hidden_state.total_prior
#
#		assert(priorA == prior), (priorA, prior)

		likelihood = 1.0
		assert(len(measurement_associations) == len(measurement_list))
		for meas_index, meas_association in enumerate(measurement_associations):
			if(meas_association == total_target_count): #birth
				likelihood *= p_birth_likelihood
			elif(meas_association == -1): #clutter
				likelihood *= p_clutter_likelihood
			else:
				assert(meas_association >= 0 and meas_association < total_target_count), (meas_association, total_target_count)
				score_index = get_score_index(score_intervals, measurement_scores[meas_index])
				likelihood *= self.memoized_assoc_likelihood(measurement_list[meas_index], meas_source_index, \
											   				 meas_association, MEAS_NOISE_COVS[meas_source_index][score_index], score_index)

		assert(prior*likelihood != 0.0), (prior, likelihood)

		return prior*likelihood

	def memoized_assoc_likelihood(self, measurement, meas_source_index, target_index, meas_noise_cov, score_index):
		"""
			LSVM and regionlets produced two measurements with the same locations (centers), so using the 
			meas_source_index as part of the key is (sort of) necessary.  Currently also using the score_index, 
			could possibly be removed (not sure if this would improve speed).

			Currently saving more in the value than necessary (from debugging), can eliminate to improve
			performance (possibly noticable)
		"""


		global CACHED_LIKELIHOODS
		global NOT_CACHED_LIKELIHOODS
		if USE_CONSTANT_R:
			if((measurement[0], measurement[1], target_index, meas_source_index, score_index) in self.assoc_likelihood_cache):
				CACHED_LIKELIHOODS = CACHED_LIKELIHOODS + 1
				return self.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, meas_source_index, score_index)]
			else:
				NOT_CACHED_LIKELIHOODS = NOT_CACHED_LIKELIHOODS + 1
				target = self.targets.living_targets[target_index]
				S = np.dot(np.dot(H, target.P), H.T) + R_default
				assert(target.x.shape == (4, 1))
		
				state_mean_meas_space = np.dot(H, target.x)
				#print type(state_mean_meas_space)
				#print state_mean_meas_space
				state_mean_meas_space = np.squeeze(state_mean_meas_space)

				if USE_PYTHON_GAUSSIAN:
					distribution = multivariate_normal(mean=state_mean_meas_space, cov=S)
					assoc_likelihood = distribution.pdf(measurement)
				else:

#					S_det = np.linalg.det(S)
					S_det = S[0][0]*S[1][1] - S[0][1]*S[1][0] # a little faster
					S_inv = inv(S)
					LIKELIHOOD_DISTR_NORM = 1.0/math.sqrt((2*math.pi)**2*S_det)

					offset = measurement - state_mean_meas_space
					a = -.5*np.dot(np.dot(offset, S_inv), offset)
					assoc_likelihood = LIKELIHOOD_DISTR_NORM*math.exp(a)



				self.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, meas_source_index, score_index)] = assoc_likelihood
				return assoc_likelihood

		else:
			if((measurement[0], measurement[1], target_index, meas_source_index, score_index) in self.assoc_likelihood_cache):
#			if((measurement[0], measurement[1], target_index, score_index) in self.assoc_likelihood_cache):
				CACHED_LIKELIHOODS = CACHED_LIKELIHOODS + 1
#				return self.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, score_index)]
#				(assoc_likelihood, cached_score_index)	= self.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, score_index)]
				(assoc_likelihood, cached_score_index, cached_measurement, cached_meas_source_index) = self.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, meas_source_index, score_index)]
				assert(cached_score_index == score_index), (cached_score_index, score_index, measurement, cached_measurement, target_index, meas_noise_cov, cached_meas_source_index, meas_source_index)
				assert(cached_meas_source_index == meas_source_index), (cached_score_index, score_index, measurement, cached_measurement, target_index, meas_noise_cov, cached_meas_source_index, meas_source_index)
#				if(cached_score_index != score_index):
#					print (cached_score_index, score_index, measurement, cached_measurement, target_index, meas_noise_cov)
#					time.sleep(2)
				return assoc_likelihood
			else:
				NOT_CACHED_LIKELIHOODS = NOT_CACHED_LIKELIHOODS + 1
				target = self.targets.living_targets[target_index]
				S = np.dot(np.dot(H, target.P), H.T) + meas_noise_cov
				assert(target.x.shape == (4, 1))
		
				state_mean_meas_space = np.dot(H, target.x)
				#print type(state_mean_meas_space)
				#print state_mean_meas_space
				state_mean_meas_space = np.squeeze(state_mean_meas_space)
				if USE_PYTHON_GAUSSIAN:
					distribution = multivariate_normal(mean=state_mean_meas_space, cov=S)
					assoc_likelihood = distribution.pdf(measurement)
				else:

##					S_det = np.linalg.det(S)

					S_det = S[0][0]*S[1][1] - S[0][1]*S[1][0] # a little faster
					S_inv = inv(S)
					LIKELIHOOD_DISTR_NORM = 1.0/math.sqrt((2*math.pi)**2*S_det)

					offset = measurement - state_mean_meas_space
					a = -.5*np.dot(np.dot(offset, S_inv), offset)
					assoc_likelihood = LIKELIHOOD_DISTR_NORM*math.exp(a)

#				self.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, score_index)] = assoc_likelihood
				self.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, meas_source_index, score_index)] = (assoc_likelihood, score_index, measurement, meas_source_index)
				return assoc_likelihood

	def debug_target_creation(self):
		print
		print "Particle ", self.id_, "importance distribution:"
		print "pi_birth = ", self.pi_birth_debug, "pi_clutter = ", self.pi_clutter_debug, \
			"pi_targets = ", self.pi_targets_debug
		print "sampled association c = ", self.c_debug, "importance reweighting factor = ", self.imprt_re_weight_debug
		self.plot_all_target_locations()

	def process_meas_assoc(self, birth_value, meas_source_index, measurement_associations, measurements, \
		widths, heights, measurement_scores, cur_time):
		"""
		- meas_source_index: the index of the measurement source being processed (i.e. in SCORE_INTERVALS)

		"""
		for meas_index, meas_assoc in enumerate(measurement_associations):
			#create new target
			if(meas_assoc == birth_value):
				self.create_new_target(measurements[meas_index], widths[meas_index], heights[meas_index], cur_time)
				new_target = True 
			#update the target corresponding to the association we have sampled
			elif((meas_assoc >= 0) and (meas_assoc < birth_value)):
				assert(meas_source_index >= 0 and meas_source_index < len(SCORE_INTERVALS)), (meas_source_index, len(SCORE_INTERVALS), SCORE_INTERVALS)
				assert(meas_index >= 0 and meas_index < len(measurement_scores)), (meas_index, len(measurement_scores), measurement_scores)
				if not (MAX_1_MEAS_UPDATE and self.targets.living_targets[meas_assoc].updated_this_time_instance):
					score_index = get_score_index(SCORE_INTERVALS[meas_source_index], measurement_scores[meas_index])
					self.targets.living_targets[meas_assoc].kf_update(measurements[meas_index], widths[meas_index], \
									heights[meas_index], cur_time, MEAS_NOISE_COVS[meas_source_index][score_index])
			else:
				#otherwise the measurement was associated with clutter
				assert(meas_assoc == -1), ("meas_assoc = ", meas_assoc)

	#@profile
	def update_particle_with_measurement(self, cur_time, measurement_lists, widths, heights, measurement_scores):
		"""
		Input:
		- measurement_lists: a list where measurement_lists[i] is a list of all measurements from the current
			time instance from the ith measurement source (i.e. different object detection algorithms
			or different sensors)
		- measurement_scores: a list where measurement_scores[i] is a list containing scores for every measurement in
			measurement_list[i]
		
		-widths: a list where widths[i] is a list of bounding box widths for the corresponding measurements
		-heights: a list where heights[i] is a list of bounding box heights for the corresponding measurements

		Debugging output:
		- new_target: True if a new target was created
		"""
		new_target = False #debugging

		birth_value = self.targets.living_count

		if InfLoopDEBUG:
			with open(debugInfLoopFile, "a") as myfile:
				myfile.write("location 1\n")

		(measurement_associations, dead_target_indices, imprt_re_weight) = \
			self.sample_data_assoc_and_death_mult_meas_per_time_proposal_distr_1(measurement_lists, \
				cur_time, measurement_scores)
		assert(len(measurement_associations) == len(measurement_lists))
		assert(imprt_re_weight != 0.0), imprt_re_weight
		self.importance_weight *= imprt_re_weight #update particle's importance weight
		if InfLoopDEBUG:
			with open(debugInfLoopFile, "a") as myfile:
				myfile.write("location 2\n")
		#process measurement associations
		for meas_source_index in range(len(measurement_associations)):
			assert(len(measurement_associations[meas_source_index]) == len(measurement_lists[meas_source_index]) and
				   len(measurement_associations[meas_source_index]) == len(widths[meas_source_index]) and
				   len(measurement_associations[meas_source_index]) == len(heights[meas_source_index]))
			self.process_meas_assoc(birth_value, meas_source_index, measurement_associations[meas_source_index], \
				measurement_lists[meas_source_index], widths[meas_source_index], heights[meas_source_index], \
				measurement_scores[meas_source_index], cur_time)

		if InfLoopDEBUG:
			with open(debugInfLoopFile, "a") as myfile:
				myfile.write("location 3\n")

		#process target deaths
		#double check dead_target_indices is sorted
		assert(all([dead_target_indices[i] <= dead_target_indices[i+1] for i in xrange(len(dead_target_indices)-1)]))
		#important to delete larger indices first to preserve values of the remaining indices
		for index in reversed(dead_target_indices):
			self.targets.kill_target(index)

		#checking if something funny is happening
		original_num_targets = birth_value
		num_targets_born = 0
		for meas_source_index in range(len(measurement_associations)):
			num_targets_born += measurement_associations[meas_source_index].count(birth_value)
		num_targets_killed = len(dead_target_indices)
		assert(self.targets.living_count == original_num_targets + num_targets_born - num_targets_killed)
		#done checking if something funny is happening

		return new_target

	def plot_all_target_locations(self):
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		for i in range(self.targets.total_count):
			life = len(self.targets.all_targets[i].all_states) #length of current targets life 
			locations_1D =  [self.targets.all_targets[i].all_states[j][0] for j in range(life)]
			ax.plot(self.targets.all_targets[i].all_time_stamps, locations_1D,
					'-o', label='Target %d' % i)

		legend = ax.legend(loc='lower left', shadow=True)
		plt.title('Particle %d, Importance Weight = %f, unique targets = %d, #targets alive = %d' % \
			(self.id_, self.importance_weight, self.targets.total_count, self.targets.living_count)) # subplot 211 title
#		plt.show()




###########assumed that the Kalman filter prediction step has already been run for this
###########target on the current time step
###########RUN PREDICTION FOR ALL TARGETS AT THE BEGINNING OF EACH TIME STEP!!!
###########@profile
##########def assoc_likelihood(measurement, target):
##########	S = np.dot(np.dot(H, target.P), H.T) + R_default
##########	assert(target.x.shape == (4, 1))
##########
##########	state_mean_meas_space = np.dot(H, target.x)
##########
##########	distribution = multivariate_normal(mean=state_mean_meas_space, cov=S)
##########	return distribution.pdf(measurement)

def normalize_importance_weights(particle_set):
	normalization_constant = 0.0
	for particle in particle_set:
		normalization_constant += particle.importance_weight
	assert(normalization_constant != 0.0), normalization_constant
	for particle in particle_set:
		particle.importance_weight /= normalization_constant


def perform_resampling(particle_set):
	print "memory used before resampling: %d" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	assert(len(particle_set) == N_PARTICLES)
	weights = []
	for particle in particle_set:
		weights.append(particle.importance_weight)
	assert(abs(sum(weights) - 1.0) < .0000001)

	new_particles = stratified_resample(weights)
	new_particle_set = []
	for index in new_particles:
		if USE_CREATE_CHILD:
			new_particle_set.append(particle_set[index].create_child())
		else:
			new_particle_set.append(copy.deepcopy(particle_set[index]))
	del particle_set[:]
	for particle in new_particle_set:
		particle.importance_weight = 1.0/N_PARTICLES
		particle_set.append(particle)
	assert(len(particle_set) == N_PARTICLES)
	#testing
	weights = []
	for particle in particle_set:
		weights.append(particle.importance_weight)
		assert(particle.importance_weight == 1.0/N_PARTICLES)
	assert(abs(sum(weights) - 1.0) < .01), sum(weights)
	print "memory used after resampling: %d" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	#done testing

def display_target_counts(particle_set, cur_time):
	target_counts = []
	for particle in particle_set:
		target_counts.append(particle.targets.living_count)
	print target_counts

	target_counts = []
	importance_weights = []
	for particle in particle_set:
		cur_target_count = 0
		for target in particle.targets.living_targets:
			if (cur_time - target.birth_time) > min_target_age:
				cur_target_count += 1
		target_counts.append(cur_target_count)
		importance_weights.append(particle.importance_weight)
	print "targets older than ", min_target_age, "seconds: ", target_counts
	print "importance weights ", min_target_age, "filler :", importance_weights


def get_eff_num_particles(particle_set):
	n_eff = 0
	weight_sum = 0
	for particle in particle_set:
		n_eff += particle.importance_weight**2
		weight_sum += particle.importance_weight

	assert(abs(weight_sum - 1.0) < .000001), (weight_sum, n_eff)
	return 1.0/n_eff



def run_rbpf_on_targetset(target_sets, online_results_filename):
	"""
	Measurement class designed to only have 1 measurement/time instance
	Input:
	- target_sets: a list where target_sets[i] is a TargetSet containing measurements from
		the ith measurement source
	Output:
	- max_weight_target_set: TargetSet from a (could be multiple with equal weight) maximum
		importance weight particle after processing all measurements
	- number_resamplings: the number of times resampling was performed
	"""
	particle_set = []
	global NEXT_PARTICLE_ID
	for i in range(0, N_PARTICLES):
		particle_set.append(Particle(NEXT_PARTICLE_ID))
		NEXT_PARTICLE_ID += 1
	prev_time_stamp = -1


	#for displaying results
	time_stamps = []
	positions = []

	iter = 0 # for plotting only occasionally
	number_resamplings = 0

	#sanity check
	number_time_instances = len(target_sets[0].measurements)
	for target_set in target_sets:
		assert(len(target_set.measurements) == number_time_instances)


	#the particle with the maximum importance weight on the previous time instance 
	prv_max_weight_particle = None

	for time_instance_index in range(number_time_instances):

		if InfLoopDEBUG:
			with open(debugInfLoopFile, "a") as myfile:
				myfile.write('-'*80 + '\n')
				myfile.write("beginning time instance %d\n" % time_instance_index)

		time_stamp = target_sets[0].measurements[time_instance_index].time
		for target_set in target_sets:
			assert(target_set.measurements[time_instance_index].time == time_stamp)
#		measurements = measurement_set.val

		measurement_lists = []
		widths = []
		heights = []
		measurement_scores = []
		for target_set in target_sets:
			measurement_lists.append(target_set.measurements[time_instance_index].val)
			widths.append(target_set.measurements[time_instance_index].widths)
			heights.append(target_set.measurements[time_instance_index].heights)
			measurement_scores.append(target_set.measurements[time_instance_index].scores)

		print "time_stamp = ", time_stamp, "living target count in first particle = ",\
		particle_set[0].targets.living_count
		if InfLoopDEBUG:
			with open(debugInfLoopFile, "a") as myfile:
				myfile.write("about to run pre-measurement update tasks\n")

		for particle in particle_set:
			#update particle death probabilities
			if(prev_time_stamp != -1):
				particle.assoc_likelihood_cache = {} #clear likelihood cache
				#Run Kalman filter prediction for all living targets
				for target in particle.targets.living_targets:
					dt = time_stamp - prev_time_stamp
					assert(abs(dt - default_time_step) < .00000001), (dt, default_time_step)
					target.kf_predict(dt, time_stamp)
				#update particle death probabilities AFTER kf_predict so that targets that moved
				#off screen this time instance will be killed
				particle.update_target_death_probabilities(time_stamp, prev_time_stamp)
		if InfLoopDEBUG:
			with open(debugInfLoopFile, "a") as myfile:
				myfile.write("done running pre-measurement update tasks\n")

		new_target_list = [] #for debugging, list of booleans whether each particle created a new target
		pIdxDebugInfo = 0
		for particle in particle_set:
			if InfLoopDEBUG:
				with open(debugInfLoopFile, "a") as myfile:
					myfile.write("about to process particle %d\n" % pIdxDebugInfo)

			new_target = particle.update_particle_with_measurement(time_stamp, measurement_lists, widths, heights, measurement_scores)
			new_target_list.append(new_target)
			pIdxDebugInfo += 1

		print "about to normalize importance weights"
		normalize_importance_weights(particle_set)
		#debugging
		if DEBUG:
			assert(len(new_target_list) == N_PARTICLES)
			for (particle_number, new_target) in enumerate(new_target_list):
				if new_target:
					print "\n\n -------Particle %d created a new target-------" % particle_number
					for particle in particle_set:
						particle.debug_target_creation()
					plt.show()
					break
		#done debugging

#		if iter%100 == 0:
#			print iter
#			display_target_counts(particle_set, time_stamp)


		if RUN_ONLINE:
			if time_instance_index >= ONLINE_DELAY:
				#find the particle that currently has the largest importance weight

				if FIND_MAX_IMPRT_TIMES_LIKELIHOOD:
					max_weight = -1
					for particle in particle_set:
						if(particle.importance_weight*particle.likelihood_DOUBLE_CHECK_ME > max_weight):
							max_weight = particle.importance_weight*particle.likelihood_DOUBLE_CHECK_ME
					cur_max_weight_target_set = None
					cur_max_weight_particle = None
					for particle in particle_set:
						if(particle.importance_weight*particle.likelihood_DOUBLE_CHECK_ME == max_weight):
							cur_max_weight_target_set = particle.targets		
							cur_max_weight_particle = particle
					print "max weight particle id = ", cur_max_weight_particle.id_

				else:
					max_imprt_weight = -1
					for particle in particle_set:
						if(particle.importance_weight > max_imprt_weight):
							max_imprt_weight = particle.importance_weight
					cur_max_weight_target_set = None
					cur_max_weight_particle = None
					for particle in particle_set:
						if(particle.importance_weight == max_imprt_weight):
							cur_max_weight_target_set = particle.targets		
							cur_max_weight_particle = particle
					print "max weight particle id = ", cur_max_weight_particle.id_


			if prv_max_weight_particle != None and prv_max_weight_particle != cur_max_weight_particle:
#				print '-'*10
#				print "Previous max weight particle:"
#				print "q[0]target IDs before matching:",
#				for cur_target in prv_max_weight_particle.targets.living_targets_q[0][1]:
#					print cur_target.id_,
#				print
#				print "q[1]target IDs before matching:",
#				for cur_target in prv_max_weight_particle.targets.living_targets_q[1][1]:
#					print cur_target.id_,
#				print
#				print "q[2]target IDs before matching:",
#				for cur_target in prv_max_weight_particle.targets.living_targets_q[2][1]:
#					print cur_target.id_,
#				print
				print "cur target IDs before matching:",
				for cur_target in prv_max_weight_particle.targets.living_targets:
					print cur_target.id_,
				print


#				print "Current max weight particle:"
#				print "q[0]target IDs before matching:",
#				for cur_target in cur_max_weight_target_set.living_targets_q[0][1]:
#					print cur_target.id_,
#				print
#				print "q[1]target IDs before matching:",
#				for cur_target in cur_max_weight_target_set.living_targets_q[1][1]:
#					print cur_target.id_,
#				print
#				print "q[2]target IDs before matching:",
#				for cur_target in cur_max_weight_target_set.living_targets_q[2][1]:
#					print cur_target.id_,
#				print
				print "cur target IDs before matching:",
				for cur_target in cur_max_weight_target_set.living_targets:
					print cur_target.id_,
				print


				if ONLINE_DELAY == 0:
					(target_associations, duplicate_ids) = match_target_ids(cur_max_weight_target_set.living_targets,\
														   prv_max_weight_particle.targets.living_targets)
					#replace associated target IDs with the IDs from the previous maximum importance weight
					#particle for ID conistency in the online results we output
					for cur_target in cur_max_weight_target_set.living_targets:
						if cur_target.id_ in target_associations:
							cur_target.id_ = target_associations[cur_target.id_]
				elif time_instance_index >= ONLINE_DELAY:
					#print time_instance_index
					#print cur_max_weight_target_set.living_targets_q
					#print prv_max_weight_particle.targets.living_targets_q
					(target_associations, duplicate_ids) = match_target_ids(cur_max_weight_target_set.living_targets_q[0][1],\
														   prv_max_weight_particle.targets.living_targets_q[0][1])
					#replace associated target IDs with the IDs from the previous maximum importance weight
					#particle for ID conistency in the online results we output
					for q_idx in range(ONLINE_DELAY):
						for cur_target in cur_max_weight_target_set.living_targets_q[q_idx][1]:
							if cur_target.id_ in duplicate_ids:
								cur_target.id_ = duplicate_ids[cur_target.id_]
							if cur_target.id_ in target_associations:
								cur_target.id_ = target_associations[cur_target.id_]
					for cur_target in cur_max_weight_target_set.living_targets:
						if cur_target.id_ in duplicate_ids:
							cur_target.id_ = duplicate_ids[cur_target.id_]						
						if cur_target.id_ in target_associations:
							cur_target.id_ = target_associations[cur_target.id_]


#				print "q[0]target IDs after matching:",
#				for cur_target in cur_max_weight_target_set.living_targets_q[0][1]:
#					print cur_target.id_,
#				print
#				print "q[1]target IDs after matching:",
#				for cur_target in cur_max_weight_target_set.living_targets_q[1][1]:
#					print cur_target.id_,
#				print
#				print "q[2]target IDs after matching:",
#				for cur_target in cur_max_weight_target_set.living_targets_q[2][1]:
#					print cur_target.id_,
#				print
				print "cur target IDs after matching:",
				for cur_target in cur_max_weight_target_set.living_targets:
					print cur_target.id_,
				print

				print "duplicate IDs:"
				print duplicate_ids
				print "target_associations:"
				print target_associations

			if time_instance_index >= ONLINE_DELAY:
				prv_max_weight_particle = cur_max_weight_particle

			#write current time step's results to results file
			if time_instance_index >= ONLINE_DELAY:
				cur_max_weight_target_set.write_online_results(online_results_filename, time_instance_index, number_time_instances)


			if ONLINE_DELAY != 0:
				print "popped on time_instance_index", time_instance_index
				for particle in particle_set:
					particle.targets.living_targets_q.popleft()

				for particle in particle_set:
					particle.targets.living_targets_q.append((time_instance_index, copy.deepcopy(particle.targets.living_targets)))
		
		if (get_eff_num_particles(particle_set) < N_PARTICLES/RESAMPLE_RATIO):
			if InfLoopDEBUG:
				with open(debugInfLoopFile, "a") as myfile:
					myfile.write("about to resample\n")

			perform_resampling(particle_set)
			print "resampled on iter: ", iter
			number_resamplings += 1
		prev_time_stamp = time_stamp

		iter+=1
		print "finished the time step"

	max_imprt_weight = -1
	for particle in particle_set:
		if(particle.importance_weight > max_imprt_weight):
			max_imprt_weight = particle.importance_weight
	max_weight_target_set = None
	for particle in particle_set:
		if(particle.importance_weight == max_imprt_weight):
			max_weight_target_set = particle.targets

	run_info = [number_resamplings]
	return (max_weight_target_set, run_info, number_resamplings)


def test_read_write_data_KITTI(target_set):
	"""
	Measurement class designed to only have 1 measurement/time instance
	Input:
	- target_set: generated TargetSet containing generated measurements and ground truth
	Output:
	- max_weight_target_set: TargetSet from a (could be multiple with equal weight) maximum
		importance weight particle after processing all measurements
	"""
	output_target_set = TargetSet()

	for measurement_set in target_set.measurements:
		time_stamp = measurement_set.time
		measurements = measurement_set.val
		widths = measurement_set.widths
		heights = measurement_set.heights

		for i in range(len(measurements)):
			output_target_set.create_new_target(measurements[i], widths[i], heights[i], time_stamp)

	return output_target_set



def convert_to_clearmetrics_dictionary(target_set, all_time_stamps):
	"""
	Convert the locations of a TargetSet to clearmetrics dictionary format

	Input:
	- target_set: TargetSet to be converted

	Output:
	- target_dict: Converted locations in clearmetrics dictionary format
	"""
	target_dict = {}
	for target in target_set.all_targets:
		for t in all_time_stamps:
			if target == target_set.all_targets[0]: #this is the first target
				if t in target.all_time_stamps: #target exists at this time
					target_dict[t] = [target.all_states[target.all_time_stamps.index(t)]]
				else: #target doesn't exit at this time
					target_dict[t] = [None]
			else: #this isn't the first target
				if t in target.all_time_stamps: #target exists at this time
					target_dict[t].append(target.all_states[target.all_time_stamps.index(t)])
				else: #target doesn't exit at this time
					target_dict[t].append(None)
	return target_dict

def calc_tracking_performance(ground_truth_ts, estimated_ts):
	"""
	!!I think clearmetrics calculates #mismatches incorrectly, look into more!!
	(has to do with whether a measurement can be mismatched to a target that doesn't exist at the current time)

	Calculate MOTA and MOTP ("Evaluating Multiple Object Tracking Performance:
	The CLEAR MOT Metrics", K. Bernardin and R. Stiefelhagen)

	Inputs:
	- ground_truth_ts: TargetSet containing ground truth target locations
	- estimated_ts: TargetSet containing esimated target locations
	"""

	#convert TargetSets to dictionary format for calling clearmetrics

	all_time_stamps = [ground_truth_ts.measurements[i].time for i in range(len(ground_truth_ts.measurements))]
	ground_truth = convert_to_clearmetrics_dictionary(ground_truth_ts, all_time_stamps)
	estimated_tracks = convert_to_clearmetrics_dictionary(estimated_ts, all_time_stamps)

	clear = clearmetrics.ClearMetrics(ground_truth, estimated_tracks, MAX_ASSOCIATION_DIST)
	clear.match_sequence()
	evaluation = [clear.get_mota(),
	              clear.get_motp(),
	              clear.get_fn_count(),
	              clear.get_fp_count(),
	              clear.get_mismatches_count(),
	              clear.get_object_count(),
	              clear.get_matches_count()]
	print 'MOTA, MOTP, FN, FP, mismatches, objects, matches'
	print evaluation     
	ground_truth_ts.plot_all_target_locations("Ground Truth")         
	ground_truth_ts.plot_generated_measurements()    
	estimated_ts.plot_all_target_locations("Estimated Tracks")      
	plt.show()

class KittiTarget:
	"""
	Used for computing target associations when outputing online results and the particle with
	the largest importance weight changes

	Values:
	- x1: smaller x coordinate of bounding box
	- x2: larger x coordinate of bounding box
	- y1: smaller y coordinate of bounding box
	- y2: larger y coordinate of bounding box
	"""
	def __init__(self, x1, x2, y1, y2):
		self.x1 = x1
		self.x2 = x2
		self.y1 = y1
		self.y2 = y2

def boxoverlap(a,b):
    """
    	Copied from  KITTI devkit_tracking/python/evaluate_tracking.py

        boxoverlap computes intersection over union for bbox a and b in KITTI format.
        If the criterion is 'union', overlap = (a inter b) / a union b).
        If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
    """
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    
    w = x2-x1
    h = y2-y1

    if w<=0. or h<=0.:
        return 0.
    inter = w*h
    aarea = (a.x2-a.x1) * (a.y2-a.y1)
    barea = (b.x2-b.x1) * (b.y2-b.y1)
    # intersection over union overlap
    o = inter / float(aarea+barea-inter)
    return o

def convert_targets(input_targets):
	kitti_format_targets = []
	for cur_target in input_targets:
		x_pos = cur_target.x[0][0]
		y_pos = cur_target.x[2][0]
		width = cur_target.width
		height = cur_target.height

		left = x_pos - width/2.0
		top = y_pos - height/2.0
		right = x_pos + width/2.0
		bottom = y_pos + height/2.0		

		kitti_format_targets.append(KittiTarget(left, right, top, bottom))
	return kitti_format_targets

def match_target_ids(particle1_targets, particle2_targets):
	"""
	Use the same association as in  KITTI devkit_tracking/python/evaluate_tracking.py

	Inputs:
	- particle1_targets: a list of targets from particle1
	- particle2_targets: a list of targets from particle2

	Output:
	- associations: a dictionary of associations between targets in particle1 and particle2.  
		associations[particle1_targetID] = particle2_targetID where particle1_targetID and
		particle2_targetID are IDs of associated targets
	"""
	kitti_targets1 = convert_targets(particle1_targets)
	kitti_targets2 = convert_targets(particle2_targets)

	#if any targets in particle1 have the same ID as a target in particle2,
	#assign the particle1 target a new ID
	duplicate_ids = {}
	global NEXT_TARGET_ID
	p2_target_ids = []
	for cur_t2 in particle2_targets:
		p2_target_ids.append(cur_t2.id_)
	for cur_t1 in particle1_targets:
		if cur_t1.id_ in p2_target_ids:
			duplicate_ids[cur_t1.id_] = NEXT_TARGET_ID
			cur_t1.id_ = NEXT_TARGET_ID
			NEXT_TARGET_ID += 1

	hm = Munkres()
	max_cost = 1e9
	cost_matrix = []
	for cur_t1 in kitti_targets1:
		cost_row = []
		for cur_t2 in kitti_targets2:
			# overlap == 1 is cost ==0
			c = 1-boxoverlap(cur_t1,cur_t2)
			# gating for boxoverlap
			if c<=.5:
				cost_row.append(c)
			else:
				cost_row.append(max_cost)
		cost_matrix.append(cost_row)

	if len(kitti_targets1) is 0:
		cost_matrix=[[]]

	# associate
	association_matrix = hm.compute(cost_matrix)
	associations = {}
	for row,col in association_matrix:
		c = cost_matrix[row][col]
		if c < max_cost:
			associations[particle1_targets[row].id_] = particle2_targets[col].id_

	return (associations, duplicate_ids)


if __name__ == "__main__":
	
	NEXT_PARTICLE_ID = 0
	if RUN_ONLINE:
		NEXT_TARGET_ID = 0 #all targets have unique IDs, even if they are in different particles
	
	# check for correct number of arguments. if user_sha and email are not supplied,
	# no notification email is sent (this option is used for auto-updates)
	if len(sys.argv)!=10:
		print "Supply 9 arguments: the number of particles (int), include_ignored_gt (bool), include_dontcare_in_gt (bool),"
		print "use_regionlets_and_lsvm (bool), sort_dets_on_intervals (bool), run_idx, total_runs, seq_idx, peripheral"
		print "received ", len(sys.argv), " arguments"
		for i in range(len(sys.argv)):
			print sys.argv[i]
		sys.exit(1);

	N_PARTICLES = int(sys.argv[1])
	run_idx = int(sys.argv[6]) #the index of this run
	total_runs = int(sys.argv[7]) #the total number of runs, for checking whether all runs are finished and results should be evaluated
	seq_idx = int(sys.argv[8]) #the index of the sequence to process
	peripheral = sys.argv[9] #should we run setup, evaluation, or an actual run?

	if not peripheral in ['setup', 'evaluate', 'run', 'standalone']:
		print "unexpected peripheral argument"
		sys.exit(1);
	else:
		print "peripheral = ", peripheral

	for i in range(2,6):
		if(sys.argv[i] != 'True' and sys.argv[i] != 'False'):
			print "Booleans must be supplied as 'True' or 'False' (without quotes)"
			sys.exit(1);


	#Should ignored ground truth objects be included when calculating probabilities? (double check specifics)
	include_ignored_gt = (sys.argv[2] == 'True')
	include_dontcare_in_gt = (sys.argv[3] == 'True')
	use_regionlets_and_lsvm = (sys.argv[4] == 'True')
	sort_dets_on_intervals = (sys.argv[5] == 'True')
#	use_regionlets = (sys.argv[10] == 'True')
	use_regionlets = None


	DESCRIPTION_OF_RUN = get_description_of_run(include_ignored_gt, include_dontcare_in_gt, 
						   use_regionlets_and_lsvm, sort_dets_on_intervals, use_regionlets)

	results_folder_name = '%s/%d_particles' % (DESCRIPTION_OF_RUN, N_PARTICLES)
#	results_folder = '%s/rbpf_KITTI_results_par_exec_trainAllButCurSeq_10runs_dup3/%s' % (DIRECTORY_OF_ALL_RESULTS, results_folder_name)
	results_folder = '%s/%s/%s' % (DIRECTORY_OF_ALL_RESULTS, CUR_EXPERIMENT_BATCH_NAME, results_folder_name)

	filename_mapping = DATA_PATH + "/evaluate_tracking.seqmap"
	n_frames         = []
	sequence_name    = []
	with open(filename_mapping, "r") as fh:
	    for i,l in enumerate(fh):
	        fields = l.split(" ")
	        sequence_name.append("%04d" % int(fields[0]))
	        n_frames.append(int(fields[3]) - int(fields[2]))
	fh.close() 
	print n_frames
	print sequence_name     
	assert(len(n_frames) == len(sequence_name))

	if peripheral == 'setup': #create directories
		print 'begin setup'
		for cur_run_idx in range(1, total_runs + 1):
			for cur_seq_idx in SEQUENCES_TO_PROCESS:
				cur_dir = '%s/results_by_run/run_%d/%s.txt' % (results_folder, cur_run_idx, sequence_name[cur_seq_idx])
				if not os.path.exists(os.path.dirname(cur_dir)):
					try:
						os.makedirs(os.path.dirname(cur_dir))
					except OSError as exc: # Guard against race condition
						if exc.errno != errno.EEXIST:
							raise
		print 'end setup'
		sys.exit(0);

	elif peripheral == 'evaluate': #evaluate results from all runs/sequences
		print 'begin evaluate'
		#make sure all the runs are complete
		all_runs_complete = True
		for cur_run_idx in range(1, total_runs + 1):
			for cur_seq_idx in SEQUENCES_TO_PROCESS:
				cur_run_complete_filename = '%s/results_by_run/run_%d/seq_%d_done.txt' % (results_folder, cur_run_idx, cur_seq_idx)
				if (not os.path.isfile(cur_run_complete_filename)):
					all_runs_complete = False
		if all_runs_complete:
			#evaluate the results when all runs are complete
			eval_metrics_file = results_folder + '/evaluation_metrics.txt' # + operator used for string concatenation!
			stdout = sys.stdout
			sys.stdout = open(eval_metrics_file, 'w')

			runs_completed = eval_results(results_folder + "/results_by_run", SEQUENCES_TO_PROCESS) # + operateor used for string concatenation!

			print "Number of runs completed = ", runs_completed
			print "Description of run: ", DESCRIPTION_OF_RUN
			print "Cached likelihoods = ", CACHED_LIKELIHOODS
			print "not cached likelihoods = ", NOT_CACHED_LIKELIHOODS
			#print "RBPF runtime (sum of all runs) = ", t1-t0
			print "USE_CONSTANT_R = ", USE_CONSTANT_R
			print "number of particles = ", N_PARTICLES
#			print "score intervals: ", SCORE_INTERVALS
#			print "run on sequences: ", SEQUENCES_TO_PROCESS
#			print "number of particles = ", N_PARTICLES

			sys.stdout.close()
			sys.stdout = stdout

			print "Printing works normally again!"

			#evaluate each sequence independently as well:
			for cur_seq_idx in SEQUENCES_TO_PROCESS:
				#evaluate the results when all runs are complete
				eval_metrics_file = results_folder + '/evaluation_metrics_seq%s.txt' % cur_seq_idx # + operator used for string concatenation!
				stdout = sys.stdout
				sys.stdout = open(eval_metrics_file, 'w')

				runs_completed = eval_results(results_folder + "/results_by_run", [cur_seq_idx]) # + operateor used for string concatenation!

				print "Number of runs completed = ", runs_completed
				print "Description of run: ", DESCRIPTION_OF_RUN
				print "Cached likelihoods = ", CACHED_LIKELIHOODS
				print "not cached likelihoods = ", NOT_CACHED_LIKELIHOODS
				#print "RBPF runtime (sum of all runs) = ", t1-t0
				print "USE_CONSTANT_R = ", USE_CONSTANT_R
				print "number of particles = ", N_PARTICLES
	#			print "score intervals: ", SCORE_INTERVALS
	#			print "run on sequences: ", SEQUENCES_TO_PROCESS
	#			print "number of particles = ", N_PARTICLES

		else:
			#evaluate the results when all runs are complete
			eval_metrics_file = results_folder + '/evaluation_metrics.txt' # + operator used for string concatenation!
			stdout = sys.stdout
			sys.stdout = open(eval_metrics_file, 'w')

			print "Error: all runs not complete"

			sys.stdout.close()
			sys.stdout = stdout

			print "Printing works normally again!"
		print 'end evaluate'
		sys.exit(0);


	elif peripheral == 'run':
		print 'begin run'
#debug
		indicate_run_started_filename = '%s/results_by_run/run_%d/seq_%d_started.txt' % (results_folder, run_idx, seq_idx)
		run_started_f = open(indicate_run_started_filename, 'w')
		run_started_f.write("This run was started\n")
		run_started_f.close()
#end debug


		indicate_run_complete_filename = '%s/results_by_run/run_%d/seq_%d_done.txt' % (results_folder, run_idx, seq_idx)
		#if we haven't already run, run now:
		if not os.path.isfile(indicate_run_complete_filename):


			#False doesn't really make sense because when actually running without ground truth information we don't know
			#whether or not a detection is ignored, but debugging. (An ignored detection is a detection not associated with
			#a ground truth object that would be associated with a don't care ground truth object if they were included.  It 
			#can also be a neighobring object type, e.g. "van" instead of "car", but this never seems to occur in the data.
			#If this occured, it would make sense to try excluding these detections.)
			include_ignored_detections = True 

			if sort_dets_on_intervals:
				MSCNN_SCORE_INTERVALS = [float(i)*.1 for i in range(3,10)]				
				REGIONLETS_SCORE_INTERVALS = [i for i in range(2, 20)]
				LSVM_SCORE_INTERVALS = [i/2.0 for i in range(0, 6)]
		#		REGIONLETS_SCORE_INTERVALS = [i for i in range(2, 16)]
		#		LSVM_SCORE_INTERVALS = [i/2.0 for i in range(0, 6)]
			else:
				MSCNN_SCORE_INTERVALS = [.5]								
				REGIONLETS_SCORE_INTERVALS = [2]
				LSVM_SCORE_INTERVALS = [0]

			#set global variables
			#global SCORE_INTERVALS
			#global TARGET_EMISSION_PROBS
			#global CLUTTER_PROBABILITIES
			#global BIRTH_PROBABILITIES
			#global MEAS_NOISE_COVS
			#global BORDER_DEATH_PROBABILITIES
			#global NOT_BORDER_DEATH_PROBABILITIES


			#train on all training sequences, except the current sequence we are testing on
			training_sequences = [i for i in [i for i in range(21)] if i != seq_idx]
			#training_sequences = [i for i in SEQUENCES_TO_PROCESS if i != seq_idx]
			#training_sequences = [0]

			#use regionlets and lsvm detections
			if use_regionlets_and_lsvm:
				SCORE_INTERVALS = [REGIONLETS_SCORE_INTERVALS, LSVM_SCORE_INTERVALS]
				(measurementTargetSetsBySequence, TARGET_EMISSION_PROBS, CLUTTER_PROBABILITIES, BIRTH_PROBABILITIES,\
					MEAS_NOISE_COVS, BORDER_DEATH_PROBABILITIES, NOT_BORDER_DEATH_PROBABILITIES) = \
						get_meas_target_sets_lsvm_and_regionlets(training_sequences, REGIONLETS_SCORE_INTERVALS, \
						LSVM_SCORE_INTERVALS, obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True,\
						include_ignored_gt = include_ignored_gt, include_dontcare_in_gt = include_dontcare_in_gt, \
						include_ignored_detections = include_ignored_detections)

			#only use regionlets detections
			else: 
				SCORE_INTERVALS = [REGIONLETS_SCORE_INTERVALS]
				(measurementTargetSetsBySequence, TARGET_EMISSION_PROBS, CLUTTER_PROBABILITIES, BIRTH_PROBABILITIES,\
					MEAS_NOISE_COVS, BORDER_DEATH_PROBABILITIES, NOT_BORDER_DEATH_PROBABILITIES) = \
					get_meas_target_sets_regionlets_general_format(training_sequences, REGIONLETS_SCORE_INTERVALS, \
					obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True, \
					include_ignored_gt = include_ignored_gt, include_dontcare_in_gt = include_dontcare_in_gt, \
					include_ignored_detections = include_ignored_detections)

			#use mscnn and regionlets detections
#			if use_regionlets:
#				SCORE_INTERVALS = [MSCNN_SCORE_INTERVALS, REGIONLETS_SCORE_INTERVALS]
#				(measurementTargetSetsBySequence, TARGET_EMISSION_PROBS, CLUTTER_PROBABILITIES, BIRTH_PROBABILITIES,\
#					MEAS_NOISE_COVS, BORDER_DEATH_PROBABILITIES, NOT_BORDER_DEATH_PROBABILITIES) = \
#						get_meas_target_sets_mscnn_and_regionlets(training_sequences, MSCNN_SCORE_INTERVALS, \
#						REGIONLETS_SCORE_INTERVALS, obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True,\
#						include_ignored_gt = include_ignored_gt, include_dontcare_in_gt = include_dontcare_in_gt, \
#						include_ignored_detections = include_ignored_detections)
#
#			#only use mscnn detections
#			else: 
#				SCORE_INTERVALS = [MSCNN_SCORE_INTERVALS]
#				(measurementTargetSetsBySequence, TARGET_EMISSION_PROBS, CLUTTER_PROBABILITIES, BIRTH_PROBABILITIES,\
#					MEAS_NOISE_COVS, BORDER_DEATH_PROBABILITIES, NOT_BORDER_DEATH_PROBABILITIES) = \
#					get_meas_target_sets_mscnn_general_format(training_sequences, MSCNN_SCORE_INTERVALS, \
#					obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True, \
#					include_ignored_gt = include_ignored_gt, include_dontcare_in_gt = include_dontcare_in_gt, \
#					include_ignored_detections = include_ignored_detections)
#
			#print "BORDER_DEATH_PROBABILITIES =", BORDER_DEATH_PROBABILITIES
			#print "NOT_BORDER_DEATH_PROBABILITIES =", NOT_BORDER_DEATH_PROBABILITIES

			#sleep(5)

			#BORDER_DEATH_PROBABILITIES = [-99, 0.3290203327171904, 0.5868263473053892, 0.48148148148148145, 0.4375, 0.42424242424242425]
			#NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.05133928571428571, 0.006134969325153374, 0.03468208092485549, 0.025735294117647058, 0.037037037037037035]

			#BORDER_DEATH_PROBABILITIES = [-99, 0.059085841694537344, 0.3982102908277405, 0.38953488372093026, 0.3611111111111111, 0.4722222222222222]
			#NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.0009339793357071974, 0.006880733944954129, 0.023255813953488372, 0.0481283422459893, 0.006944444444444444]

			assert(len(n_frames) == len(measurementTargetSetsBySequence))
		#	############DEBUG
		#	
		#	print "target emission probs: "
		#	print TARGET_EMISSION_PROBS
		#	print "cluter probs: "
		#	print CLUTTER_PROBABILITIES
		#	print "birth probs: "
		#	print BIRTH_PROBABILITIES
		#	print "Meas noise covs:"
		#	print MEAS_NOISE_COVS
		#	print "BORDER_DEATH_PROBABILITIES:"
		#	print BORDER_DEATH_PROBABILITIES
		#	print "NOT_BORDER_DEATH_PROBABILITIES:"
		#	print NOT_BORDER_DEATH_PROBABILITIES
		#	sleep(5)
		#	##########DONE DEBUG

			t0 = time.time()
			info_by_run = [] #list of info from each run
			cur_run_info = None
		################	for seq_idx in SEQUENCES_TO_PROCESS:
			results_filename = '%s/results_by_run/run_%d/%s.txt' % (results_folder, run_idx, sequence_name[seq_idx])

			debugInfLoopFile = '%s/results_by_run/run_%d/debugInfLoop.txt' % (results_folder, run_idx)	



			print "Processing sequence: ", seq_idx
			tA = time.time()
			(estimated_ts, cur_seq_info, number_resamplings) = run_rbpf_on_targetset(measurementTargetSetsBySequence[seq_idx], results_filename)
			#cProfile.run('run_rbpf_on_targetset(measurementTargetSetsBySequence[seq_idx], results_filename)')
			print "done processing sequence: ", seq_idx
			
			tB = time.time()
			this_seq_run_time = tB - tA
			cur_seq_info.append(this_seq_run_time)
			if cur_run_info == None:
				cur_run_info = cur_seq_info
			else:
				assert(len(cur_run_info) == len(cur_seq_info))
				for info_idx in len(cur_run_info):
					#assuming for now info can be summed over each sequence in a run!
					#works for runtime and number of times resampling is performed
					cur_run_info[info_idx] += cur_seq_info[info_idx]

			print "about to write results"

			if not RUN_ONLINE:
				estimated_ts.write_targets_to_KITTI_format(num_frames = n_frames[seq_idx], filename = results_filename)
			print "done write results"
			print "running the rbpf took %f seconds" % (tB-tA)
		################END	for seq_idx in SEQUENCES_TO_PROCESS:
			
			info_by_run.append(cur_run_info)
			t1 = time.time()

			stdout = sys.stdout
			sys.stdout = open(indicate_run_complete_filename, 'w')

			print "This run is finished (and this file indicates the fact)\n"
			print "Resampling was performed %d times\n" % number_resamplings
			print "This run took %f seconds\n" % (t1-t0)

			print "TARGET_EMISSION_PROBS=", TARGET_EMISSION_PROBS
			print "CLUTTER_PROBABILITIES=", CLUTTER_PROBABILITIES
			print "BIRTH_PROBABILITIES=", BIRTH_PROBABILITIES
			print "MEAS_NOISE_COVS=", MEAS_NOISE_COVS
			print "BORDER_DEATH_PROBABILITIES=", BORDER_DEATH_PROBABILITIES
			print "NOT_BORDER_DEATH_PROBABILITIES=", NOT_BORDER_DEATH_PROBABILITIES


			sys.stdout.close()
			sys.stdout = stdout


		print 'end run'
		sys.exit(0);

	else: #peripheral == 'standalone'

		print 'begin standalone run'

		#False doesn't really make sense because when actually running without ground truth information we don't know
		#whether or not a detection is ignored, but debugging. (An ignored detection is a detection not associated with
		#a ground truth object that would be associated with a don't care ground truth object if they were included.  It 
		#can also be a neighobring object type, e.g. "van" instead of "car", but this never seems to occur in the data.
		#If this occured, it would make sense to try excluding these detections.)
		include_ignored_detections = True 

		if sort_dets_on_intervals:
			MSCNN_SCORE_INTERVALS = [float(i)*.1 for i in range(5,10)]
			REGIONLETS_SCORE_INTERVALS = [i for i in range(2, 20)]
			LSVM_SCORE_INTERVALS = [i/2.0 for i in range(0, 6)]
	#		REGIONLETS_SCORE_INTERVALS = [i for i in range(2, 16)]
	#		LSVM_SCORE_INTERVALS = [i/2.0 for i in range(0, 6)]
		else:
			MSCNN_SCORE_INTERVALS = [.5]				
			REGIONLETS_SCORE_INTERVALS = [2]
			LSVM_SCORE_INTERVALS = [0]

		#set global variables
		#global SCORE_INTERVALS
		#global TARGET_EMISSION_PROBS
		#global CLUTTER_PROBABILITIES
		#global BIRTH_PROBABILITIES
		#global MEAS_NOISE_COVS
		#global BORDER_DEATH_PROBABILITIES
		#global NOT_BORDER_DEATH_PROBABILITIES


		#train on all training sequences, except the current sequence we are testing on
		#training_sequences = [i for i in [i for i in range(21)] if i != seq_idx]
		training_sequences = [i for i in range(21)]
		#training_sequences = [i for i in SEQUENCES_TO_PROCESS if i != seq_idx]
		#training_sequences = [0]

		#use regionlets and lsvm detections
		if use_regionlets_and_lsvm:
			SCORE_INTERVALS = [REGIONLETS_SCORE_INTERVALS, LSVM_SCORE_INTERVALS]
			(measurementTargetSetsBySequence, TARGET_EMISSION_PROBS, CLUTTER_PROBABILITIES, BIRTH_PROBABILITIES,\
				MEAS_NOISE_COVS, BORDER_DEATH_PROBABILITIES, NOT_BORDER_DEATH_PROBABILITIES) = \
					get_meas_target_sets_lsvm_and_regionlets(training_sequences, REGIONLETS_SCORE_INTERVALS, \
					LSVM_SCORE_INTERVALS, obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True,\
					include_ignored_gt = include_ignored_gt, include_dontcare_in_gt = include_dontcare_in_gt, \
					include_ignored_detections = include_ignored_detections)

		#only use regionlets detections
		else: 
			SCORE_INTERVALS = [REGIONLETS_SCORE_INTERVALS]
			(measurementTargetSetsBySequence, TARGET_EMISSION_PROBS, CLUTTER_PROBABILITIES, BIRTH_PROBABILITIES,\
				MEAS_NOISE_COVS, BORDER_DEATH_PROBABILITIES, NOT_BORDER_DEATH_PROBABILITIES) = \
				get_meas_target_sets_regionlets_general_format(training_sequences, REGIONLETS_SCORE_INTERVALS, \
				obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True, \
				include_ignored_gt = include_ignored_gt, include_dontcare_in_gt = include_dontcare_in_gt, \
				include_ignored_detections = include_ignored_detections)

		print "BORDER_DEATH_PROBABILITIES =", BORDER_DEATH_PROBABILITIES
		print "NOT_BORDER_DEATH_PROBABILITIES =", NOT_BORDER_DEATH_PROBABILITIES

		#sleep(5)

		#BORDER_DEATH_PROBABILITIES = [-99, 0.3290203327171904, 0.5868263473053892, 0.48148148148148145, 0.4375, 0.42424242424242425]
		#NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.05133928571428571, 0.006134969325153374, 0.03468208092485549, 0.025735294117647058, 0.037037037037037035]

		#BORDER_DEATH_PROBABILITIES = [-99, 0.059085841694537344, 0.3982102908277405, 0.38953488372093026, 0.3611111111111111, 0.4722222222222222]
		#NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.0009339793357071974, 0.006880733944954129, 0.023255813953488372, 0.0481283422459893, 0.006944444444444444]

		assert(len(n_frames) == len(measurementTargetSetsBySequence))
	#	############DEBUG
	#	
	#	print "target emission probs: "
	#	print TARGET_EMISSION_PROBS
	#	print "cluter probs: "
	#	print CLUTTER_PROBABILITIES
	#	print "birth probs: "
	#	print BIRTH_PROBABILITIES
	#	print "Meas noise covs:"
	#	print MEAS_NOISE_COVS
	#	print "BORDER_DEATH_PROBABILITIES:"
	#	print BORDER_DEATH_PROBABILITIES
	#	print "NOT_BORDER_DEATH_PROBABILITIES:"
	#	print NOT_BORDER_DEATH_PROBABILITIES
	#	sleep(5)
	#	##########DONE DEBUG

		t0 = time.time()
		info_by_run = [] #list of info from each run
		cur_run_info = None
	################	for seq_idx in SEQUENCES_TO_PROCESS:
		filename = './temp_standalone_results.txt'

		print "Processing sequence: ", seq_idx
		tA = time.time()
		(estimated_ts, cur_seq_info, number_resamplings) = run_rbpf_on_targetset(measurementTargetSetsBySequence[seq_idx], results_filename)
		#estimated_ts = cProfile.run('run_rbpf_on_targetset(measurementTargetSetsBySequence[seq_idx], results_filename)')
		print "done processing sequence: ", seq_idx
		
		tB = time.time()
		this_seq_run_time = tB - tA
		cur_seq_info.append(this_seq_run_time)
		if cur_run_info == None:
			cur_run_info = cur_seq_info
		else:
			assert(len(cur_run_info) == len(cur_seq_info))
			for info_idx in len(cur_run_info):
				#assuming for now info can be summed over each sequence in a run!
				#works for runtime and number of times resampling is performed
				cur_run_info[info_idx] += cur_seq_info[info_idx]

		print "about to write results"
		estimated_ts.write_targets_to_KITTI_format(num_frames = n_frames[seq_idx], filename = filename)
		print "done write results"
		print "running the rbpf took %f seconds" % (tB-tA)
	################END	for seq_idx in SEQUENCES_TO_PROCESS:
		
		info_by_run.append(cur_run_info)
		t1 = time.time()


		print "Resampling was performed %d times\n" % number_resamplings
		print "This run took %f seconds\n" % (t1-t0)

		print 'end run'
		sys.exit(0);



