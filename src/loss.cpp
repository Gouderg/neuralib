#include "../header/loss.hpp"

double Loss::calculate(const TensorInline& output, const TensorInline& y) {

    std::vector<double> samples_losses = forward(output, y);
    
    return std::reduce(samples_losses.begin(), samples_losses.end()) / samples_losses.size();
}

double Loss::regularization_loss(const Layer_Dense& layer) {
    
    double regularization_loss = 0.0;

    TensorInline w = layer.getWeights();
    TensorInline b = layer.getBiases();

    if (layer.getWeightRegL1() > 0) {
        regularization_loss += layer.getWeightRegL1() * TensorInline::sum(w.abs());
    }

    if (layer.getWeightRegL2() > 0) {
        regularization_loss += layer.getWeightRegL2() * TensorInline::sum(w * w);
    }

    if (layer.getBiasRegL1() > 0) {
        regularization_loss += layer.getBiasRegL1() * TensorInline::sum(b.abs());
    }

    if (layer.getBiasRegL2() > 0) {
        regularization_loss += layer.getBiasRegL2() * TensorInline::sum(b * b);
    }
    return regularization_loss;
}

std::vector<double> Loss::forward(const TensorInline &y_pred, const TensorInline& y_true) {
    std::vector<double> a(y_true.getHeight(), 0.0);
    return a;
}

double Loss::accuracy(const TensorInline &inputs, const TensorInline &y) {
    
    // Get the indice of the best score.
    std::vector<int> predictions;
    for (int i = 0; i < inputs.getHeight() * inputs.getWidth(); i += inputs.getWidth()) {
        predictions.push_back(std::max_element(inputs.tensor.begin() + i , inputs.tensor.begin() + i + inputs.getWidth()) - (inputs.tensor.begin() + i));
    }

    // get the indice of the best ground truth.
    std::vector<double> y_flat;
    if (y.getHeight() == 2 ) {
        for (int i = 0; i < y.getHeight() * y.getWidth(); i += y.getWidth()) {
            y_flat.push_back(std::max_element(y.tensor.begin() + i , y.tensor.begin() + i + y.getWidth()) - (y.tensor.begin() + i));
        }
    } else {
        y_flat = y.tensor;
    }

    // Compute the mean.
    double somme = 0.0;
    for (int i = 0; i < static_cast<int>(predictions.size()); i++) {
        if (predictions[i] == y_flat[i]) {
            somme += 1.0;
        }
    }

    return somme / predictions.size();
}