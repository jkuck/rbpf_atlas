import random
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCALED = True
ONE_MEAS_PER_TIME = False
WITH_NOISE = True
NOISE_SD = .05 #Standard deviation of noise, before scaling
NUM_GEN_FRAMES = 500
TEST_SHORT_SEGMENT = False

ADD_CLUTTER = True
#Clutter count for a given time step will be drawn from a Poisson distribution with expectation CLUTTER_LAMBDA
CLUTTER_LAMBDA = 10 
SHUFFLE_MEASUREMENTS = True

prob_detection = 0.95

default_time_step = 0.1 

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

        self.measurements = [] #generated measurements for a generative TargetSet 




def gen_data(measurement_plot_filename):
    returnTargetSet = TargetSet()

    x1_list = []
    y1_list = []
    x2_list = []
    y2_list = []
    x3_list = []
    y3_list = []
    x4_list = []
    y4_list = []
    all_x = []
    all_y = []
    for i in range(0,NUM_GEN_FRAMES):
        print "i =", i

        if WITH_NOISE:

            if SCALED:
#                x1 = 600*(-math.sin(2*math.pi*i/500) + NOISE_SD*np.random.normal(0, 1)) + 800
#                y1 = 180*(-math.cos(2*math.pi*i/500) + NOISE_SD*np.random.normal(0, 1)) + 200
#
#                x2 = 600*(math.sin(math.pi*i/500 - math.pi/2) + NOISE_SD*np.random.normal(0, 1)) + 800
#                y2 = 180*(-1.0 + math.cos(math.pi*i/500 - math.pi/2) + NOISE_SD*np.random.normal(0, 1)) + 200
#
#                x3 = 600*(0.0 + NOISE_SD*np.random.normal(0, 1)) + 800
#                y3 = 180*(1.0 - 2.0*i/500 + NOISE_SD*np.random.normal(0, 1)) + 200
#
#                x4 = 600*(1.0 - 2.0*i/500 + NOISE_SD*np.random.normal(0, 1)) + 800
#                y4 = 180*(0.0 + NOISE_SD*np.random.normal(0, 1)) + 200

                x1 = 300*(2*(-math.sin(2*math.pi*i/500)) + NOISE_SD*np.random.normal(0, 1)) + 800
                y1 = 90*(2*(-math.cos(2*math.pi*i/500)) + NOISE_SD*np.random.normal(0, 1)) + 200
    
                x2 = 300*(2*(math.sin(math.pi*i/500 - math.pi/2)) + NOISE_SD*np.random.normal(0, 1)) + 800
                y2 = 90*(2*(-1.0 + math.cos(math.pi*i/500 - math.pi/2)) + NOISE_SD*np.random.normal(0, 1)) + 200
    
                x3 = 300*(2*(0.0) + NOISE_SD*np.random.normal(0, 1)) + 800
                y3 = 90*(2*(1.0 - 2.0*i/500) + NOISE_SD*np.random.normal(0, 1)) + 200
                
                x4 = 300*(2*(1.0 - 2.0*i/500) + NOISE_SD*np.random.normal(0, 1)) + 800
                y4 = 90*(2*(0.0) + NOISE_SD*np.random.normal(0, 1)) + 200

            else:
                x1 = (-math.sin(2*math.pi*i/500)) + NOISE_SD*np.random.normal(0, 1)
                y1 = (-math.cos(2*math.pi*i/500)) + NOISE_SD*np.random.normal(0, 1)
    
                x2 = (math.sin(math.pi*i/500 - math.pi/2)) + NOISE_SD*np.random.normal(0, 1)
                y2 = (-1.0 + math.cos(math.pi*i/500 - math.pi/2)) + NOISE_SD*np.random.normal(0, 1)
    
                x3 = (0.0) + NOISE_SD*np.random.normal(0, 1)
                y3 = (1.0 - 2.0*i/500) + NOISE_SD*np.random.normal(0, 1)
                
                x4 = (1.0 - 2.0*i/500) + NOISE_SD*np.random.normal(0, 1)
                y4 = (0.0) + NOISE_SD*np.random.normal(0, 1)
        else:
            x1 = (-math.sin(2*math.pi*i/500))
            y1 = (-math.cos(2*math.pi*i/500))

            x2 = (math.sin(math.pi*i/500 - math.pi/2))
            y2 = (-1.0 + math.cos(math.pi*i/500 - math.pi/2))

            x3 = (0)
            y3 = (1.0 - 2.0*i/500)

            x4 = (1 - 2.0*i/500)
            y4 = (0)   


        cur_meas = Measurement()

        if i > 100 and i < 400 and random.random() < prob_detection:
            cur_meas.val.append(np.array([x1, y1]))

        if i < 300 and random.random() < prob_detection:
            cur_meas.val.append(np.array([x2, y2]))

        if i < 400 and random.random() < prob_detection:    
            cur_meas.val.append(np.array([x3, y3]))

        if random.random() < prob_detection:
            cur_meas.val.append(np.array([x4, y4]))

        if ADD_CLUTTER:
            clutter_count = np.random.poisson(CLUTTER_LAMBDA, 1)[0]
            x_min = -2.0
            x_max = 2.0
            y_min = -2.0
            y_max = 2.0
            if SCALED:
                x_min = 300*x_min + 800
                x_max = 300*x_max + 800
                y_min = 90*y_min + 200
                y_max = 90*y_max + 200
            for j in range(clutter_count):
                x_clutter = random.uniform(x_min, x_max)
                y_clutter = random.uniform(y_min, y_max)
                cur_meas.val.append(np.array([x_clutter, y_clutter]))

        if(SHUFFLE_MEASUREMENTS):
            np.random.shuffle(cur_meas.val)

        if(ONE_MEAS_PER_TIME):
            cur_meas.val = [random.choice(cur_meas.val)]

        if(TEST_SHORT_SEGMENT and (i < 100 or i > 105)):
            cur_meas.val = []

        for j in range(len(cur_meas.val)):
            all_x.append(cur_meas.val[j][0])
            all_y.append(cur_meas.val[j][1])
        cur_meas.widths = [1 for j in range(len(cur_meas.val))]
        cur_meas.heights = [1 for j in range(len(cur_meas.val))]
        cur_meas.scores = [3 for j in range(len(cur_meas.val))]
        cur_meas.time = i*default_time_step

        returnTargetSet.measurements.append(cur_meas)

    #plot measurements
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(all_x, all_y, '+', label='Target %d' % 1)    
#    ax.plot(x1_list, y1_list, '+', label='Target %d' % 1)
#    ax.plot(x2_list, y2_list, '+', label='Target %d' % 2)
#    ax.plot(x3_list, y3_list, '+', label='Target %d' % 3)
#    ax.plot(x4_list, y4_list, '+', label='Target %d' % 4)

#           legend = ax.legend(loc='lower left', shadow=True)
#           plt.title('%s, unique targets = %d, #targets alive = %d' % \
#               (title, self.total_count, self.living_count)) # subplot 211 title   
    fig.savefig(measurement_plot_filename)  


    return returnTargetSet



