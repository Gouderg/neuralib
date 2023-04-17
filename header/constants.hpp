#ifndef CONSTANTS_H
#define CONSTANTS_H

// Dataset.
const int NB_POINT = 2;
const int NB_REGRESSION_POINT = 1000;
const int NB_LABEL_CATEGORICAL = 3;
const int NB_LABEL_BINARY = 2;
const int NB_INPUTS = 2;

// Size of the network.
const int NB_EPOCH = 10000;
const int NB_NEURON = 64;

// Uniform distribution parameters.
const double MEAN = 0.0;
const double STD_DEVIATION = 1.0;

// Optimizer.
const double LEARNING_RATE = 0.005;
const double DECAY = 1e-3;
const double MOMENTUM_EPSILON = 0.0; 

// Regularization Loss.
const double WEIGHT_L1 = 0.0;
const double WEIGHT_L2 = 5e-4;
const double BIAS_L1 = 0.0;
const double BIAS_L2 = 5e-4;

// Dropout.
const double DROPOUT_RATE = 0.1;

// Accuracy.
const double STRICT_ACCURACY_METRICS = 250.0;

#endif