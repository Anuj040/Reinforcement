"""Base configuration class"""

import numpy as np

class Config(object):
    ######Model Parameters
    MODEL = 'Hierarchy'
    CLASS_OBJECT = 1 #'Aeroplane'
    CLASSES = 20#Total number of PASCAL CLASSES
    VISUAL_DESCRIPTOR = 25088

    SCALE_SUBREGION = float(3)/4#sub-region scale for hierarchical regions
    SCALE_MASK = float(1)/(SCALE_SUBREGION*4)

    #Image Attributes
    IMG_ROWS = 224
    IMG_COLS = 224
    IMG_CHNS = 3

    ######Vizualization Parameters
    DRAW = 0.05 #Percentage of samples to be drawn

    ####Dataset
    TWO_DATABASES = 0#Boolean to indicate if two databases to be used

    #Experience/History Parameters
    HIS_STEPS = 4

    #State & RL Parameters
    ACTIONS = 6
    IOU_THRESHOLD = 0.5
    REWARD_TERMINAL = 3
    REWARD_MOVEMENT = 1

    ######Training Parameters
    MAX_STEPS = 10#Max steps for object detection
    GAMMA = 0.9#DISCOUNT FACTOR FOR REWARD
    EPSILON = 1.0 #GREEDY POLICY 
    BATCH_SIZE = 100
    REPLAY_BUFFER = 1000#Experiences to store in memory
    LEARNING_RATE = 1e-6


    def __init__(self):
        self.IMG_SHAPE = np.array([self.IMG_ROWS, self.IMG_COLS,
                            self.IMG_CHNS])
                            






