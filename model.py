import os
import sys
import datetime
import re
from PIL import Image, ImageDraw


import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import numpy as np
    import keras
    import keras.layers as KL
    import keras.models as KM
    from keras.initializers import random_normal
    from keras.utils import generic_utils
import random

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
from GAN.style_transfer_clean.VGG16 import VGG16
from Reinforcement.image_assist import load_images_in_data_set, get_all_images, load_image_labels
from Reinforcement.parse_xml_annotations import get_bb_gt, generate_bounding_box
from Reinforcement.metrics import calculate_overlapping, follow_iou
from Reinforcement.visualization import draw_sequences, draw_sequences_test
from Reinforcement.reinforcement import get_state, get_reward_trigger, get_reward_movement, update_history_vector

#Path to PASCAL VOC 2012 datset
PATH_VOC = os.path.join(ROOT_DIR, 'dataset/VOC2012')
#Path to VGG Weights
PATH_VGG = os.path.join(ROOT_DIR, 'logs/VGG16/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
#Path for storing visualization for search sequences
PATH_TEST = './Output_1'
PATH_TEST_INFER = './Output_infer'


#Path to PASCAL VOC test datset
PATH_VOC_TEST = os.path.join(ROOT_DIR, 'dataset/VOC2007')
if not os.path.exists(PATH_TEST):
    os.makedirs(PATH_TEST)
if not os.path.exists(PATH_TEST_INFER):
    os.makedirs(PATH_TEST_INFER)

class Hierarchy_Object_Detect:
    """Model class for Hierarchical Object Detection
        with Deep Reinforcement Learning [arXiv:1611.03718v2 [cs.CV] 25 Nov 2016]
    """

    def __init__(self, config, mode, model_dir, run = 0):
        
        self.config = config
        self.mode = mode
        self.run = run
        self.model_dir = model_dir
        self.set_log_dir()
        self.build(config)

    def build(self, config):
        #Models
        z = KL.Input(shape = config.IMG_SHAPE, name = 'VGG_input')
        VGG = VGG16(z, include_top = True)
        self.VGG = KM.Model(inputs = z, outputs = VGG, name = 'vgg16')
        self.VGG.load_weights(PATH_VGG)
        self.QNET_model = self.QNET(str(config.CLASS_OBJECT))

    def QNET(self, name = None):
        input_shape = (self.config.VISUAL_DESCRIPTOR + self.config.ACTIONS*self.config.HIS_STEPS,)
        input_tensor = KL.Input(shape = input_shape, name = name + '_Q_input')
        x = KL.Dense(1024, activation = 'relu', kernel_initializer=random_normal(stddev=0.01), name = name + '_Dense_1')(input_tensor)
        x = KL.Dropout(0.2)(x)
        x = KL.Dense(1024, activation = 'relu', kernel_initializer=random_normal(stddev=0.01), name = name + '_Dense_2')(x)
        x = KL.Dropout(0.2)(x)
        x = KL.Dense(self.config.ACTIONS, activation = 'linear', kernel_initializer=random_normal(stddev=0.01), name = name + '_Action')(x)
        
        return KM.Model(inputs = input_tensor, outputs = x, name = name + '_QNET')

    def datagen(self, config):
        """Prepare data"""
        #Lodaing image names
        if self.mode == 'inference':
            self.image_names = np.array([load_images_in_data_set('aeroplane_test', PATH_VOC_TEST)])
        else:
            self.image_names = np.array([load_images_in_data_set('trainval', PATH_VOC)])
        #Loading Image files
        if self.mode == 'inference':
            self.images = get_all_images(self.image_names, PATH_VOC_TEST)
        else:
            self.images = get_all_images(self.image_names, PATH_VOC)
        
        if self.mode == 'inference':
            self.labels = load_image_labels(self.image_names, 'aeroplane_test', PATH_VOC_TEST)
        
        if self.mode == 'train':
            self.annotation = []
            self.classes_gt_objects = []
            for j in range(np.size(self.image_names)):
                Annotate = get_bb_gt(self.image_names[0][j], PATH_VOC)
                self.annotation.append(Annotate)
                self.classes_gt_objects.append(Annotate[:, 0])#Object classes for the respective annotations

    def train(self, config, epochs):
        self.datagen(config)
        self.compile()
        print("Data Preparation and Model Compilation Done")
        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        h = np.zeros([config.CLASSES])#Pointer for storing last experience in experience
                                      #replay buffer. One for each category of the PASCAL
                                      #for simultaneous training of each class

        #Initialize replay memories
        replay = [[] for _ in range(config.CLASSES)]
        reward = 0#Start with a zero reward state
        BOOL_TRAIN = False #Tracking if sufficient samples in memory to commence training
        self.eps = config.EPSILON
        for i in range(self.epoch, epochs):

            progbar = generic_utils.Progbar(np.size(self.image_names))
            for j in range(np.size(self.image_names)):
                #Initialization
                masked = 0 
                unfinished = 1

                #Image and contained object Infromation
                image = np.array(self.images[j])
                image_name = self.image_names[0][j]
                gt_masks = generate_bounding_box(self.annotation[j], image.shape)
                classes_gt_objects = self.classes_gt_objects[j]

                region_mask = np.ones([image.shape[0], image.shape[1]])
                shape_gt_masks = np.shape(gt_masks)
                available_objects = np.ones([np.size(classes_gt_objects)])
                #Iterate through all objects in the ground truth of an image
                for k in range(np.size(classes_gt_objects)):

                    #Init Visualization
                    background = Image.new('RGBA', (10000, 2500), (255, 255, 255, 255))
                    draw = ImageDraw.Draw(background)

                    #Check if the GT object is of the target class category
                    if classes_gt_objects[k] == config.CLASS_OBJECT:
                        gt_mask = gt_masks[:, :, k]
                        step = 0
                        new_iou = 0
                        #Matrix for storing IoU of each object of the GT
                        last_matrix = np.zeros([np.size(classes_gt_objects)])

                        region_image = image
                        offset = [0, 0]
                        size_mask = [image.shape[0], image.shape[1]]
                        original_shape = size_mask
                        old_region_mask = region_mask
                        region_mask = np.ones([image.shape[0], image.shape[1]])

                        #In case, GT object is already masked by other already found
                        #mask, ignore it for training
                        if masked == 1:
                            for p in range(gt_masks.shape[2]):
                                overlap = calculate_overlapping(old_region_mask, gt_masks[:, :, p])
                                if overlap > 0.60:#Almost all of GT is within old_region_mask
                                        available_objects[p] = 0
                        #Checking if there still are unfound objects
                        if np.count_nonzero(available_objects) == 0:
                            unfinished = 0
                        
                        #At every time step, follow_iou function calculates the ground truth object with 
                        #max. overlap with visual region, for calculating rewards appropriately
                        iou, new_iou, last_matrix, index = follow_iou(gt_masks, region_mask, classes_gt_objects,
                                                                        config.CLASS_OBJECT, last_matrix, available_objects)
                        
                        new_iou = iou                        
                        gt_mask = gt_masks[:, :, index]#GT_Mask with the max. overlap with region_mask
                        
                        #Initialize history vector (6 past actions over 4 steps)
                        history_vector = np.zeros([config.ACTIONS*config.HIS_STEPS])

                        #Computing Initial state
                        state = get_state(config, region_image, history_vector, self.VGG)
                        
                        #Status indicates if the agent is still active and hasn't taken the terminal action
                        status = 1
                        action = 0
                        reward = 0

                        if step > config.MAX_STEPS:
                            BOOL_DRAW = False
                            if random.random() < config.DRAW:
                                BOOL_DRAW = True
                            background = draw_sequences(i, k, step, action, draw, region_image, background,
                                                        PATH_TEST, iou, reward, gt_mask, region_mask, image_name,
                                                        BOOL_DRAW)
                            step += 1
                        while (status == 1) & (step < config.MAX_STEPS) & unfinished:
                            category = int(classes_gt_objects[k] - 1)
                            qval = self.QNET_model.predict(state.T)
                            background = draw_sequences(i, k, step, action, draw, region_image, background,
                                                        PATH_TEST, iou, reward, gt_mask, region_mask, image_name,
                                                        save_boolean = False)#self.config.BOOL_DRAW)
                            step += 1
                            #Forced termination for IoU > 0.5 for faster training
                            if (i < 10) & (new_iou > config.IOU_THRESHOLD):
                                action = 6
                            #epsilon-greedy policy#Exploration
                            elif random.random() < config.EPSILON:
                                action = np.random.randint(1, 7)
                            else: 
                                action = (npargmax(qval)) + 1#Predicted action
                            
                            #Termination
                            if action == 6:
                                iou, new_iou, last_matrix, index = follow_iou(gt_masks, region_mask, classes_gt_objects,
                                                                        config.CLASS_OBJECT, last_matrix, available_objects)
                                gt_mask = gt_masks[:, :, index]
                                reward = get_reward_trigger(config, new_iou)
                                
                                BOOL_DRAW = False
                                if random.random() < config.DRAW:
                                    BOOL_DRAW = True
                                background = draw_sequences(i, k, step, action, draw, region_image, background,
                                                            PATH_TEST, iou, reward, gt_mask, region_mask, image_name,
                                                            BOOL_DRAW)
                                step +=1

                            #Movement action, corresponding sub-region is cropped
                            else:
                                region_mask = np.zeros(original_shape)
                                #Sub-region is shrunk
                                size_mask = [int(size_mask[0] * config.SCALE_SUBREGION), int(size_mask[1] * config.SCALE_SUBREGION)]
                                offset_aux, offset = self.Movement(size_mask, offset, action)

                                region_image = region_image[offset_aux[0]:offset_aux[0] + size_mask[0],
                                                            offset_aux[1]:offset_aux[1] + size_mask[1]]
                                region_mask[offset[0]:offset[0]+size_mask[0],
                                             offset[1]:offset[1]+size_mask[1]] = 1
                                
                                iou, new_iou, last_matrix, index = follow_iou(gt_masks, region_mask, classes_gt_objects,
                                                                        config.CLASS_OBJECT, last_matrix, available_objects)
                                gt_mask = gt_masks[:, :, index]#GT_Mask with the max. overlap with region_mask
                                reward = get_reward_movement(config, iou, new_iou)
                                iou = new_iou
                            history_vector = update_history_vector(config, history_vector, action)#Stores the history of last #'HIS_STEPS' actions
                            new_state = get_state(config, region_image, history_vector, self.VGG)#Feature extracted updated state of the region
                            
                            #Experience replay storage
                            if len(replay[category]) < config.REPLAY_BUFFER:
                                replay[category].append((state, action, reward, new_state))
                                if len(replay[category]) > 150: 
                                    BOOL_TRAIN = True

                            else:
                                if h[category] < (config.REPLAY_BUFFER - 1):
                                    h[category] += 1#Pointer to the buffer where latest experience has to be stored
                                else:
                                    h[category] = 0
                                h_aux = h[category]
                                h_aux = int(h_aux)
                                replay[category][h_aux] = (state, action, reward, new_state)
                                
                            state = new_state
                            if action == 6:
                                status = 0
                                masked = 1
                                #Mask the discovered object with GT for faster learning
                                image[gt_mask == 1] = 120
                            else:
                                masked = 0
                        available_objects[index] = 0 #Object Detected
                if BOOL_TRAIN & (random.random() < 0.02):#No Need to train after every memory update
                    self.train_minibatch(config, replay[category], progbar)

            if not BOOL_TRAIN:
                print("Not Enough Training Buffer.",
                        "\nIncrease the number of image samples")
                print(len(replay[category]))
                exit()

            if self.eps > 0.1:
                self.eps -= 0.1
            checkpoint = self.checkpoint_path.replace("*epoch*", "{:04d}".format(i+1))
            self.QNET_model.save_weights(checkpoint)

    def compile(self):

        sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.VGG.compile(optimizer=sgd, loss='categorical_crossentropy')
        optimizer = keras.optimizers.Adam(lr = self.config.LEARNING_RATE)
        self.QNET_model.compile(optimizer = optimizer, loss = 'mse')

    def detect(self, config):
        self.datagen(config)
        print('Running the model in Inference mode')
        for j in range(np.size(self.image_names)):
            if self.labels[j] == '1':
                image = np.array(self.images[j])

                #Initialize Visualization
                background = Image.new('RGBA', (10000, 2000), (255, 255, 255, 255))
                draw = ImageDraw.Draw(background)
                image_name = self.image_names[0][j]
                Annotate = get_bb_gt(image_name, PATH_VOC_TEST)
                gt_masks = generate_bounding_box(Annotate, image.shape)
                classes_gt_objects = Annotate[:, 0]
                size_mask = (image.shape[0], image.shape[1])
                
                
                original_shape = size_mask
                image_for_search = image
                #Offset of the observation region for each time step
                offset = (0, 0)

                Absolute_Status = True#Boolean to indicate if the agent continues to search
                                      #If the first object covers the entire image, no need
                                      #to continue searching there
                action = 0
                step = 0
                qval = 0
                region_image = image_for_search
                region_mask = np.ones(original_shape)
                BOOL_DRAW = False
                #Agent acts while max allowed steps not taken and the Boolean
                while (step < config.MAX_STEPS) and (Absolute_Status):
                    #Initialize History
                    history_vector = np.zeros([config.ACTIONS*config.HIS_STEPS])
                    state = get_state(config, region_image, history_vector, self.VGG)
                    status = 1
                    draw_sequences_test(step, action, qval, draw, region_image, background, PATH_TEST_INFER,
                    region_mask, image_name, BOOL_DRAW)

                    while (status==1) and (step < config.MAX_STEPS):
                        step += 1
                        qval = self.QNET_model.predict(state.T)
                        action = np.argmax(qval) + 1
                        #Movement action
                        if action != 6:
                            region_mask = np.zeros(original_shape)
                            size_mask = (int(size_mask[0]*config.SCALE_SUBREGION), int(size_mask[1]*config.SCALE_SUBREGION))
                            
                            offset_aux, offset = self.Movement(size_mask, offset, action)

                            region_image = region_image[offset_aux[0]:offset_aux[0] + size_mask[0],
                                                        offset_aux[1]:offset_aux[1] + size_mask[1]]
                            region_mask[offset[0]:offset[0]+size_mask[0],
                                            offset[1]:offset[1]+size_mask[1]] = 1
                        if (step == config.MAX_STEPS) or (action == 6):
                            BOOL_DRAW = True
                        draw_sequences_test(step, action, qval, draw, region_image, background, PATH_TEST_INFER,
                                            region_mask, image_name, BOOL_DRAW)

                        #Trigger action
                        if action == 6:
                            offset = (0, 0)
                            status = 0
                            if (step == 1) or (config.ONLY_FIRST_OBJECT):
                                Absolute_Status = False
                            image_for_search[region_mask == 1] = 120
                            region_image = image_for_search
                        history_vector = update_history_vector(config, history_vector, action)
                        new_state = get_state(config, region_image, history_vector, self.VGG)
                        state = new_state

    def Movement(self, size_mask, offset, action):
        if action == 1:
            offset_aux = [0, 0]
            offset = [0, 0]
        elif action == 2:
            offset_aux = [0, int(size_mask[1]*self.config.SCALE_MASK)]
            offset = [offset[0], offset[1] + int(size_mask[1]*self.config.SCALE_MASK)]
        elif action == 3:
            offset_aux = [int(size_mask[0]*self.config.SCALE_MASK), 0]
            offset = [offset[0] + int(size_mask[0]*self.config.SCALE_MASK), offset[1]]
        elif action == 4:
            offset_aux = [int(size_mask[0]*self.config.SCALE_MASK),
                            int(size_mask[1]*self.config.SCALE_MASK)]
            offset = [offset[0] + int(size_mask[0]*self.config.SCALE_MASK),
                        offset[1] + int(size_mask[1]*self.config.SCALE_MASK)]
        elif action == 5:
            offset_aux = [int(size_mask[0]*self.config.SCALE_MASK/2),
                            int(size_mask[0]*self.config.SCALE_MASK/2)]
            offset = [offset[0] + int(size_mask[0]*self.config.SCALE_MASK/2),
                        offset[1] + int(size_mask[0]*self.config.SCALE_MASK/2)]
        return offset_aux, offset

    def train_minibatch(self, config, replay, progbar):
        #Random minibatch from the saved experiences
        minibatch  = random.sample(replay, config.BATCH_SIZE)
        X_train = []
        Y_train = []
        
        #Minibatch sampled from replay memory for training
        for memory in minibatch:
            old_state, action, reward, new_state = memory
            old_qval = self.QNET_model.predict(old_state.T)
            newQ = self.QNET_model.predict(new_state.T)
            maxQ = np.max(newQ)#Max possible future reward
            y = np.zeros([1, self.config.ACTIONS])
            y = old_qval
            y = y.T
            if action != 6:#Non-terminal state
                update = reward + self.config.GAMMA*maxQ#Bellman Equation#Discounted Future reward
            else:#Terminal state
                update = reward
            y[action-1] = update#target_output
            X_train.append(old_state)
            Y_train.append(y)
        X_train = np.array(X_train).astype("float32")
        Y_train = np.array(Y_train).astype("float32")
        X_train = X_train[:, :, 0]
        Y_train = Y_train[:, :, 0]
        
        hist = self.QNET_model.fit(X_train, Y_train, batch_size = self.config.BATCH_SIZE)
        #print(hist)
        # progbar.add(n = 10, values=hist)

    def find_last(self):
        """Locates the latest model checkpoint
            Returns:
                Path to the latest checkpoint file
        """
        #Parse through directory names. Each directory referes to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.MODEL
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)

        if not dir_names:
            import errno
            raise FileNotFoundError(
                        errno.ENOENT,
                        'No model directory found at {}'.format(self.model_dir))
        #Pick latest directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])

        #Latest Checkpoint
        checkpoints = next(os.walk(dir_name))[2]

        #Finding the latest stored model 
        checkpoints = filter(lambda f: f .startswith('model_' + str(self.config.CLASS_OBJECT) + '_'), checkpoints)
        checkpoints = sorted(checkpoints)

        if not checkpoints:
            import errno
            raise FileNotFoundError(
                        errno.ENOENT,
                        'No weight files in {}'.format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath):
        """
        Custom function for loading weights
        """
        import h5py
        from keras.engine import saving

        if h5py is None:
            raise ImportError("'load_weights' requires h5py.'")
        
        f = h5py.File(filepath, mode = 'r')
        layers = self.QNET_model.layers
        saving.load_weights_from_hdf5_group(f, layers)

        if hasattr(f, 'close'):
            f.close()
        #Update the log directory
        self.set_log_dir(filepath)

    def set_log_dir(self, model_path = None):
        """Sets the model directory and the epoch counter.
        model_path: If None, or a format mismatch, then set a new directory
        and start epochs from 0. Otherwise, extract the log directory and the 
        epoch counter from the file name.       
        
        """
        #Assume a start from beginning
        self.epoch = 0
        now = datetime.datetime.now()

        #If model path with date and epoch exists, use it.
        if model_path:
            #Get epoch and date from the file name
            #Sample path for windows
            # \path\to\logs_pix2pix\facades_20200607T1304\model_1_0001.h5

            split = re.split(r'\\', model_path)
            now = [a for a in split if a.startswith(self.config.MODEL + '_')][0]#Existing model's date and time
            
            regex = r".*[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})"
            m = re.match(regex,now)

            e = [a for a in split if a.startswith('model_'+ str(self.config.CLASS_OBJECT) + '_')][0]#Last epoch
            regex = r".*[\w-]+(\d{4})\.h5"
            e = re.match(regex, e)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                                
                #Epoch No. in file 1-based, in Keras 0-based.
                #Adjust by one to start from the next epoch
                self.epoch = int(e.group(1))
                print('Resuming from epoch %d' % (self.epoch + 1))
        #Directory for train logs
        self.log_dir = os.path.join(self.model_dir, "{}_{:%Y%m%dT%H%M}".format(
                                    self.config.MODEL, now))
        print('Log:', self.log_dir)
        #Path to save after every epoch. Include placeholders to be filled for epoch
        self.checkpoint_path = os.path.join(self.log_dir, "model_" + str(self.config.CLASS_OBJECT) + "_{}_*epoch*.h5".format(
                                                            self.run))
    




