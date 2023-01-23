#include "../header/layer_dense.hpp"

// Constructor.
Layer_Dense::Layer_Dense(const int n_inputs, const int n_neurons) {
    
    // Init layers.
    this->weights = Tensor(n_inputs, n_neurons, 1);
    this->biases = Tensor(1, n_neurons);

    // Init momemtums layers with 0 with the same shape.
    this->weight_momentums = Tensor(n_inputs, n_neurons, 0);
    this->bias_momentums = Tensor(1, n_neurons, 0);
}

void Layer_Dense::forward(Tensor& inputs) {
    
    this->inputs = inputs;

    this->output = inputs.dot(this->weights) + this->biases;
}

void Layer_Dense::backward(Tensor &dvalues) {
    
    // Gradients on parameters.
    this->dweights = this->inputs.transposate().dot(dvalues);
    
    std::vector<double> somme(dvalues.shapeX(), 0);
    
    for (int i = 0; i < dvalues.shapeY(); i++) {
        for(int j = 0; j < dvalues.shapeX(); j++) {
            somme[j] += dvalues.getValue(i, j);
        }
    }

    this->dbiases = Tensor(1,dvalues.shapeX());
    this->dbiases.setRow(0, somme); 

    // Gradients on values.
    this->dinputs = dvalues.dot(this->weights.transposate());
}
