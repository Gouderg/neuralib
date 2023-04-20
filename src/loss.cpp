#include "../header/loss.hpp"

LossValues Loss::calculate(const TensorInline& output, const TensorInline& y, const bool with_regularization) {

    std::vector<double> samples_losses = this->forward(output, y);
    
    double data_loss = std::reduce(samples_losses.begin(), samples_losses.end()) / samples_losses.size();

    double regularization_loss = with_regularization ? this->regularization_loss() : 0.0;

    return { .data_loss=data_loss, .regularization_loss=regularization_loss } ;
}

double Loss::regularization_loss() {
    
    double regularization_loss = 0.0;

    TensorInline w, b;

    for (auto layer : this->trainable_layers) {

        w = layer->getWeights();
        b = layer->getBiases();
        
        if (layer->getWeightRegL1() > 0) {
            regularization_loss += layer->getWeightRegL1() * TensorInline::sum(w.abs());
        }

        if (layer->getWeightRegL2() > 0) {
            regularization_loss += layer->getWeightRegL2() * TensorInline::sum(w * w);
        }

        if (layer->getBiasRegL1() > 0) {
            regularization_loss += layer->getBiasRegL1() * TensorInline::sum(b.abs());
        }

        if (layer->getBiasRegL2() > 0) {
            regularization_loss += layer->getBiasRegL2() * TensorInline::sum(b * b);
        }
    }

    return regularization_loss;
}

std::vector<double> Loss::forward(const TensorInline &y_pred, const TensorInline& y_true) {
    std::vector<double> a(y_true.getHeight(), 0.0);
    return a;
}