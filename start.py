import os 
import sys
import argparse


ROOT_DIR = os.path.abspath('../')

#Import Project
sys.path.append(ROOT_DIR)
from Reinforcement.config import Config
import Reinforcement.model as modellib
DEFAULT_LOG_DIR = os.path.join(ROOT_DIR, 'logs')

class TrainConfig(Config):
    pass
class TestConfig(Config):
    ONLY_FIRST_OBJECT = True

def train(model, config, args):
    model.train(config, args.N)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical Object Detection')

    parser.add_argument('command', 
                        metavar="<command>",
                        help="'train' or 'inference'")
    parser.add_argument('--N', '-n', default = 60,
                        required=False, type = int,
                        metavar="No. of training epochs")
    parser.add_argument('--run', '-r', type = int,
                        default=1, help = "No. of training run")
    parser.add_argument('--logs', required = False,
                            metavar='path/to/logs',
                            default=DEFAULT_LOG_DIR)
    parser.add_argument('--weights', '-w', default = False,
                        type = str, help =  "'True' if reload weights")

    args = parser.parse_args()
    if args.command == 'train':
        config = TrainConfig()
        model = modellib.Hierarchy_Object_Detect(config, mode = 'train', model_dir = args.logs, run = args.run)
    elif args.command == 'test':
        config = TestConfig()
        model = modellib.Hierarchy_Object_Detect(config, mode = 'inference', model_dir = args.logs)

    if args.weights:
        weights_path = model.find_last()
        #Load Weights
        print('Loading Weights:', weights_path)
        model.load_weights(weights_path)
    
    if args.command == 'train':
        train(model, config, args)
    elif args.command == 'test':
        model.detect(config)
