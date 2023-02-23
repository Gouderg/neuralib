#include "../header/optimizer.hpp"

// Constructor.
Optimizer::Optimizer(const double learning_rate, const double decay, const double mom_ep){
    this->learning_rate = learning_rate;
    this->current_lr = learning_rate;
    this->decay = decay;
    this->iterations = 0;
    this->mom_ep = mom_ep;
}

// Update.
void Optimizer::pre_update_params() {
    if (this->decay) {
        this->current_lr = this->learning_rate * (1.0 / (1.0 + this->decay * this->iterations));
    }
}

void Optimizer::post_update_params() {
    this->iterations += 1;
}

void Optimizer_SGD::update_params(Layer_Dense &layer) {
    
    TensorInline w, b;
    if (this->mom_ep) {
        w = layer.getWeightMomentum() * this->mom_ep - layer.getDweights() * this->current_lr;
        layer.setWeightMomentum(w);

        b = layer.getBiasMomentum() * this->mom_ep - layer.getDbiases() * this->current_lr;
        layer.setBiasMomentum(b);

    } else {
        w = layer.getDweights() * -this->current_lr;
        b = layer.getDbiases() * -this->current_lr;
    }
    layer.addWeights(w);
    layer.addBiases(b);
}

void Optimizer_Adagrad::update_params(Layer_Dense &layer) {

    TensorInline w, b;

    w = layer.getSquaredDWeights();
    b = layer.getSquaredDBias();
    layer.addWeightCache(w);
    layer.addBiasCache(b);

    w = layer.getDweights() * -this->current_lr / (layer.getWeightCache().sqrt() + this->mom_ep);
    b = layer.getDbiases() * -this->current_lr / (layer.getBiasCache().sqrt() + this->mom_ep);

    layer.addWeights(w);
    layer.addBiases(b);
}

void Optimizer_RMSprop::update_params(Layer_Dense &layer) {

    TensorInline w, b;

    layer.setWeightCache(layer.getWeightCache() * this->rho + layer.getSquaredDWeights() * (1 - this->rho));
    layer.setBiasCache(layer.getBiasCache() * this->rho + layer.getSquaredDBias() * (1 - this->rho));

    w = layer.getDweights() * -this->current_lr / (layer.getWeightCache().sqrt() + this->mom_ep);
    b = layer.getDbiases() * -this->current_lr / (layer.getBiasCache().sqrt() + this->mom_ep);
    layer.addWeights(w);
    layer.addBiases(b);
}

void Optimizer_Adam::update_params(Layer_Dense &layer) {

    TensorInline w, b, w_mom, b_mom, w_cache, b_cache;

    layer.setWeightMomentum(layer.getWeightMomentum() * this->beta1 + layer.getDweights() * (1 - this->beta1));
    layer.setBiasMomentum(layer.getBiasMomentum() * this->beta1 + layer.getDbiases() * (1 - this->beta1));

    w_mom = layer.getWeightMomentum() / (1.0 - pow(this->beta1, this->iterations + 1.0));
    b_mom = layer.getBiasMomentum() / (1.0 - pow(this->beta1, this->iterations + 1.0));

    layer.setWeightCache(layer.getWeightCache() * this->beta2 + layer.getSquaredDWeights() * (1 - this->beta2));
    layer.setBiasCache(layer.getBiasCache() * this->beta2 + layer.getSquaredDBias() * (1 - this->beta2));

    w_cache = layer.getWeightCache() / (1.0 - pow(this->beta2, this->iterations + 1.0));
    b_cache = layer.getBiasCache() / (1.0 - pow(this->beta2, this->iterations + 1.0));

    w = w_mom * (-this->current_lr) / (w_cache.sqrt() + this->mom_ep);
    b = b_mom * (-this->current_lr) / (b_cache.sqrt() + this->mom_ep);

    layer.addWeights(w);
    layer.addBiases(b);
}

std::ostream& operator <<(std::ostream& out, const Optimizer&) {
    out << "Optimizer de base tout sec sans rien sans sucre sans sel ";
    return out;
}

std::ostream& operator <<(std::ostream& out, const Optimizer_SGD&) {
    out << "Optimizer SGD ";
    return out;
}

std::ostream& operator <<(std::ostream& out, const Optimizer_Adagrad&) {
    out << "Optimizer Adagrad ";
    return out;
}

std::ostream& operator <<(std::ostream& out, const Optimizer_RMSprop&) {
    out << "Optimizer RMSprop ";
    return out;
}

std::ostream& operator <<(std::ostream& out, const Optimizer_Adam&) {
    out << "Optimizer Adam ";
    return out;
}