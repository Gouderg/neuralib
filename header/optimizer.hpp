#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>

#include "layer_dense.hpp"
#include "tensor.hpp"

class Optimizer_SGD {

    public: 
        Optimizer_SGD(double learning_rate = 1.0);

        void update_params(Layer_Dense &layer);

    private:
        double learning_rate;
};

#endif