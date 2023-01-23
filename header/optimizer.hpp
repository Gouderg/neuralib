#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>

#include "layer_dense.hpp"
#include "tensor.hpp"

class Optimizer_SGD {

    public: 
        Optimizer_SGD(const double learning_rate = 1.0, const double decay = 0.0, const double momemtum = 0.0);

        // Getter.
        double getLr() const { return this->learning_rate; }
        double getCurrentLr() const { return this->current_lr; }

        
        // Update.
        void pre_update_params();
        void update_params(Layer_Dense &layer);
        void post_update_params();

    private:
        double learning_rate, current_lr, decay, momemtum;
        int iterations;

};

#endif