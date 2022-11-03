#include "../header/layer_dense.hpp"

// Constructor.
Layer_Dense::Layer_Dense(const int n_inputs, const int n_neurons) {
    this->weights = Tensor(n_inputs, n_neurons, 1);
    this->biases = Tensor(1, n_neurons);

}

void Layer_Dense::forward(Tensor& inputs) {
    
    this->inputs = inputs;

    this->output = inputs.dot(this->weights) + this->biases;
}

void Layer_Dense::backward(Tensor &dvalues) {
    
    // Gradients on parameters.
    this->dweights = this->inputs.transposate().dot(dvalues);
    // this->biases = 

    // Gradients on values.
    this->dinputs = dvalues.dot(this->weights.transposate());
}
