#include "../header/optimizer.hpp"

// Constructor.
Optimizer_SGD::Optimizer_SGD(double learning_rate) {
    this->learning_rate = learning_rate;
}

// Update.
void Optimizer_SGD::update_params(Layer_Dense &layer) {
    Tensor w = layer.getDweights() * -this->learning_rate;
    Tensor b = layer.getDbiases() * -this->learning_rate;
    layer.addWeights(w);
    layer.addBiases(b);
}