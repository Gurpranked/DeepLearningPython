# Gurpreet Singh
# 1/15/2023
# Machine Learning
# Professor Geggis

# This program uses runtime args instead of hard coded values for each run, making development easier, faster, and closer to reality in industry.

import mnist_loader
import network
import numpy as np
import argparse

# A: Represents the nueron count within the second layer of the 3 layer network (arg.Hyperparameter[0])
# B: Represents the nueron count withint eh third and final layer of the 3 layer network, the output layer, must be 10 to match the output dimension. (arg.Hyperparameter[1])
# X: Represents the epochs for which to train the data (arg.Hyperparameter[2])
# Y: Represents the mini-batch size for each epoch of training, typically in multiple of 8 for computational efficiency (arg.Hyperparameter[3])
# Z: Represents the learning rate for the model. The learning rate represents the step the model is to 
#    take each epoch in which direction to get closer to target value (arg.Hyperparameter[4])


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('Hyperparameters', metavar ='N', 
                    type = int, nargs = 5,
                    help ='Sequence of hyperparameters for A, B, X, Y, Z, in that order')

    return parser

args = process_args().parse_args()

training_data, validation_data, testing_data = mnist_loader.load_data_wrapper()
net = network.Network([784, args.Hyperparameters[0], args.Hyperparameters[1]])

net.SGD(training_data, args.Hyperparameters[2], args.Hyperparameters[3], args.Hyperparameters[4], testing_data)
