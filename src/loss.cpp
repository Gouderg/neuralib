#include "../header/loss.hpp"

double Loss::calculate(const TensorInline& output, const TensorInline& y) {

    std::vector<double> samples_losses = forward(output, y);

    return std::reduce(samples_losses.begin(), samples_losses.end()) / samples_losses.size();
}

double Loss::regularization_loss(const Layer_Dense& layer) {
    
    double regularization_loss = 0;

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
    std::vector<double> a(y_true.getHeight(), 0);
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
            somme += 1;
        }
    }

    return somme / predictions.size();
}

std::vector<double> Loss_CategoricalCrossEntropy::forward(const TensorInline &y_pred, const TensorInline& y_true) {

    // Clip data to prevent division by 0.
    TensorInline y_pred_clipped = y_pred;

    for (int i = 0; i < y_pred_clipped.getHeight() * y_pred_clipped.getWidth(); i++) {        
        if ((y_pred_clipped.tensor[i] < 1e-15)) {
            y_pred_clipped.tensor[i] = 1e-15;
        }

        if ((y_pred_clipped.tensor[i] > 1 - 1e-15)) {
            y_pred_clipped.tensor[i] = 1 - 1e-15;
        }
        
    }

    // Probabilities for target values only if categoricals values.
    std::vector<double> correct_confidences(y_true.getWidth(), 0);

    if (y_true.getHeight() == 1) {
        for (int i = 0; i < y_true.getWidth(); i++) {
            correct_confidences[i] = y_pred_clipped.tensor[i * y_pred_clipped.getWidth() + y_true.tensor[i]];
        }
    } else {
        for (int i = 0; i < y_true.getHeight() * y_true.getWidth(); i++) {
            correct_confidences[i] += y_pred_clipped.tensor[i] * y_true.tensor[i];
        }
    }

    // Losses.
    for (int i = 0; i < static_cast<int>(correct_confidences.size()); i++) {
        correct_confidences[i] = -log(correct_confidences[i]);
    }

    return correct_confidences;
}

void Loss_CategoricalCrossEntropy::backward(const TensorInline &dvalues, const TensorInline &y_true) {

    int samples = dvalues.getHeight();
    int labels = dvalues.getWidth();
    this->dinputs = TensorInline({samples, labels});

    TensorInline y_flat_diag({samples, labels});
    if (y_true.getHeight() == 1) {
        for (int i = 0; i < samples; i ++) {
            y_flat_diag.tensor[i * y_flat_diag.getWidth() + y_true.tensor[i]] = 1;
        }
    } else {
        y_flat_diag = y_true;
    }

    for (int i = 0; i < samples * labels; i ++) {
        this->dinputs.tensor[i] = (-y_flat_diag.tensor[i] / dvalues.tensor[i] / samples);
    }
}