#include "../header/layer_dropout.hpp"

Layer_Dropout::Layer_Dropout(const double rate) {
    this->rate = 1.0 - rate; // Success rate, for a dropout of 0.1 we need a success rate of 0.9.
}


void Layer_Dropout::forward(const TensorInline &inputs, const bool training) {
    this->inputs = inputs;
    if (!training) {
        this->output = inputs;
        return;
    }
    // Generate and save scale mask.
    this->binary_mask = TensorInline::binomial({1, this->rate, inputs.getHeight(), inputs.getWidth()}) / this->rate;
    
    // Apply mask.
    this->output = inputs * this->binary_mask;
}

void Layer_Dropout::backward(const TensorInline &dvalues) {
    this->dinputs = dvalues * this->binary_mask;
}