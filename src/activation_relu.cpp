#include "../header/activation_relu.hpp"

void Activation_ReLU::forward(const TensorInline &inputs, const bool training) {
    this->inputs = inputs;

    this->output = TensorInline({inputs.getHeight(), inputs.getWidth()});
    
    for (int i = 0; i < inputs.getHeight() * inputs.getWidth(); i++) {
            this->output.tensor[i] = inputs.tensor[i] > 0 ? inputs.tensor[i] : 0.0;
    }
}

void Activation_ReLU::backward(const TensorInline &dvalues) {
    this->dinputs = dvalues;

    for (int i = 0; i < this->dinputs.getHeight() * this->dinputs.getWidth(); i++) {
        if (this->inputs.tensor[i] <= 0) {
            this->dinputs.tensor[i] = 0.0;
        }
    }
}