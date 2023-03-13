#include "../header/activation_linear.hpp"

void Activation_Linear::forward(const TensorInline &inputs) {
    this->inputs = inputs;
    this->output = inputs;
}

void Activation_Linear::backward(const TensorInline &dvalues) {
    this->dinputs = dvalues;
}