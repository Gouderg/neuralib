#include "../header/activation_sigmoid.hpp"

void Activation_Sigmoid::forward(const TensorInline & inputs, const bool training) {
    this->inputs = inputs;
    this->output = 1.0 / (1.0 + TensorInline::exp(-1.0 * inputs));
}

void Activation_Sigmoid::backward(const TensorInline & dvalues) {
    this->dinputs = TensorInline({dvalues.getHeight(), dvalues.getWidth()});
    this->dinputs = dvalues * (1.0 - this->output) * this->output;
}