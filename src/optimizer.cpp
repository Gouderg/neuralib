#include "../header/optimizer.hpp"

// Constructor.
Optimizer_SGD::Optimizer_SGD(const double learning_rate, const double decay, const double momemtum){
    this->learning_rate = learning_rate;
    this->current_lr = learning_rate;
    this->decay = decay;
    this->iterations = 0;
    this->momemtum = momemtum;
}

// Update.

void Optimizer_SGD::pre_update_params() {
    if (this->decay) {
        this->current_lr = this->learning_rate * (1.0 / (1.0 + this->decay * this->iterations));
    }
}

void Optimizer_SGD::update_params(Layer_Dense &layer) {
    
    Tensor w, b;
    if (this->momemtum) {
        w = layer.getWeightMomentums() * this->momemtum - layer.getDweights() * this->current_lr;
        layer.setWeightMomemtums(w);

        b = layer.getBiasMomentums() * this->momemtum - layer.getDbiases() * this->current_lr;
        layer.setBiasMomemtums(b);
    } else {
        w = layer.getDweights() * -this->current_lr;
        b = layer.getDbiases() * -this->current_lr;
    }
    layer.addWeights(w);
    layer.addBiases(b);
}

void Optimizer_SGD::post_update_params() {
    this->iterations += 1;
}