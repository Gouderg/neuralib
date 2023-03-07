#include "../header/layer_dropout.hpp"

Layer_Dropout::Layer_Dropout(const double rate) {
    this->rate = 1.0 - rate; // Success rate.
}


void Layer_Dropout::forward(const TensorInline &inputs) {
    this->inputs = inputs;
    try {
        this->binary_mask = TensorInline::binomial({1, this->rate, inputs.getHeight(), inputs.getWidth()}) / this->rate;
    } catch (std::invalid_argument & error) {
        std::cout << "Error: Layer_Dropout::forward - " << error.what() << std::endl;
        exit(1);
    }
    this->output = inputs * this->binary_mask;
}

void Layer_Dropout::backward(const TensorInline &dvalues) {
    this->dinputs = dvalues * this->binary_mask;
}