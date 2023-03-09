#ifndef CONSTANTS_H
#define CONSTANTS_H

// Dataset.
const int NB_POINT = 100;
const int NB_LABEL = 2;
const int NB_INPUTS = 2;

// Size of the network.
const int NB_EPOCH = 1000;
const int NB_NEURON = 128;

// Uniform distribution parameters.
const double MEAN = 0.0;
const double STD_DEVIATION = 1.0;

// Optimizer.
const double LEARNING_RATE = 0.001;
const double DECAY = 5e-7;
const double MOMENTUM_EPSILON = 0.0; 

// Regularization Loss.
const double WEIGHT_L1 = 0.0;
const double WEIGHT_L2 = 5e-4;
const double BIAS_L1 = 0.0;
const double BIAS_L2 = 5e-4;

// Dropout.
const double DROPOUT_RATE = 0.1;

#endif