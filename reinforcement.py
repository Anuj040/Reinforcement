import os
import sys

import numpy as np
import cv2
import keras.backend as K 


ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)


def get_state(config, image, history_vector, vgg_model):
    descriptor_image = get_conv_image_descriptor_for_image(config, image, vgg_model)#Feature Extracted version of input image
    descriptor_image = np.reshape(descriptor_image, (config.VISUAL_DESCRIPTOR, 1))
    history_vector = np.reshape(history_vector, (config.ACTIONS * config.HIS_STEPS, 1))
    state = np.vstack((descriptor_image, history_vector))
    return state

def get_conv_image_descriptor_for_image(config, image, model):
    im = cv2.resize(image, (config.IMG_ROWS, config.IMG_COLS)).astype(np.float32)
    im = np.expand_dims(im, axis = 0)

    _convout1_f = K.function(
                [model.layers[0].input, K.learning_phase()],
                [model.layers[-7].output]#'block5_pool' layer
                )
    return _convout1_f([im, 0])

def get_reward_trigger(config, new_iou):
    if new_iou >= config.IOU_THRESHOLD:
        reward = config.REWARD_TERMINAL
    else:
        reward = -config.REWARD_TERMINAL
    return reward

def get_reward_movement(config, iou, new_iou):
    if new_iou < iou:
        reward = -config.REWARD_MOVEMENT
    else:
        reward = config.REWARD_MOVEMENT#As the movement step happens after shrinkage of ROIs, maintaining IoU is rewarded
    return reward

def update_history_vector(config, history_vector, action):
    """Updates the stored history of last #'HIS_STEPS' actions
        Args:
            history_vector: Stored hitory to be updated
            action: Latest Action
        Returns:
            updated action history
            """
    #Initialize
    action_vector = np.zeros(config.ACTIONS)
    size_history_vector = np.size(np.nonzero(history_vector))#No of steps taken so far.
    updated_history_vector = np.zeros(config.ACTIONS*config.HIS_STEPS)

    action_vector[action-1] = 1
    if size_history_vector < config.HIS_STEPS:
        aux2 = config.ACTIONS*size_history_vector 
        history_vector[aux2:aux2+config.ACTIONS] = action_vector
        return history_vector
    else:
        updated_history_vector[:config.ACTIONS*(config.HIS_STEPS - 1)] = history_vector[config.ACTIONS:]
        updated_history_vector[config.ACTIONS*(config.HIS_STEPS - 1):] = action_vector
        return updated_history_vector
