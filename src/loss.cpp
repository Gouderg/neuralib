#include "../header/loss.hpp"

double Loss::calculate(Tensor& output, Tensor& y) {

    std::vector<double> samples_losses = forward(output, y);

    return std::reduce(samples_losses.begin(), samples_losses.end()) / samples_losses.size();
}

double Loss::regularization_loss(Layer_Dense& layer) {
    
    double regularization_loss = 0;

    Tensor w = layer.getWeights();
    Tensor b = layer.getBiases();

    if (layer.getWeightRegL1() > 0) {
        regularization_loss += layer.getWeightRegL1() * Tensor::sum(w.abs());
    }

    if (layer.getWeightRegL2() > 0) {
        regularization_loss += layer.getWeightRegL2() * Tensor::sum(w * w);
    }

    if (layer.getBiasRegL1() > 0) {
        regularization_loss += layer.getBiasRegL1() * Tensor::sum(b.abs());
    }

    if (layer.getBiasRegL2() > 0) {
        regularization_loss += layer.getBiasRegL2() * Tensor::sum(b * b);
    }
    return regularization_loss;

}

std::vector<double> Loss::forward(Tensor& y_pred, Tensor& y_true) {
    std::vector<double> a(y_true.shapeY(), 0);
    return a;
}

double Loss::accuracy(Tensor &inputs, Tensor y) {
    
    // Get the indice of the best score.
    std::vector<int> predictions;
    for (auto &row : inputs.getTensor()) {
        predictions.push_back(std::max_element(row.begin(),row.end()) - row.begin());
    }

    // get the indice of the best ground truth.
    std::vector<double> y_flat;
    if (y.shapeY() == 1) {
        y_flat = y.getRow(0);
    } else {
        for (auto &row : y.getTensor()) {
            y_flat.push_back(std::max_element(row.begin(),row.end()) - row.begin());
        }
    }
    
    // Compute the mean.
    double somme = 0.0;
    for (int i = 0; i < predictions.size(); i++) {
        if (predictions[i] == y_flat[i]) {
            somme += 1;
        }
    }

    return somme / predictions.size();
}

std::vector<double> Loss_CategoricalCrossEntropy::forward(Tensor& y_pred, Tensor& y_true) {

    // Clip data to prevent division by 0.
    std::vector<std::vector<double>> y_pred_clipped = y_pred.getTensor();

    for (int i = 0; i < y_pred_clipped.size(); i++) {
        for (int j = 0; j < y_pred_clipped[0].size(); j++) {
            if ((y_pred_clipped[i][j] < 1e-15)) {
                y_pred_clipped[i][j] = 1e-15;
            }

            if ((y_pred_clipped[i][j] > 1 - 1e-15)) {
                y_pred_clipped[i][j] = 1 - 1e-15;
            }
        }
    }

    // Probabilities for target values only if categoricals values.
    std::vector<double> correct_confidences(y_true.shapeX(), 0);

    if (y_true.shapeY() == 1) {
        for (int i = 0; i < y_true.shapeX(); i++) {
            correct_confidences[i] = y_pred_clipped[i][y_true.getValue(0, i)];
        }
    } else {
        for (int i = 0; i < y_true.shapeX(); i++) {
            for (int j = 0; j < y_true.shapeY(); j++) {
                correct_confidences[i] += y_pred_clipped[i][j] * y_true.getValue(i, j);
            }
        }
    }

    // Losses.
    for (int i = 0; i < correct_confidences.size(); i++) {
        correct_confidences[i] = -log(correct_confidences[i]);
    }

    return correct_confidences;
}

void Loss_CategoricalCrossEntropy::backward(Tensor &dvalues, Tensor &y_true) {

    int samples = dvalues.shapeY();
    int labels = dvalues.shapeX();
    this->dinputs = Tensor(samples, labels);

    Tensor y_flat_diag(samples, labels);
    if (y_true.shapeY() == 1) {
        for (int i = 0; i < samples; i ++) {
            y_flat_diag.setValue(i, y_true.getValue(0, i), 1);
        }
    } else {
        y_flat_diag = y_true;
    }
    for (int i = 0; i < samples; i ++) {
        for (int j = 0; j < labels; j ++) {
            this->dinputs.setValue(i, j, (-y_flat_diag.getValue(i, j) / dvalues.getValue(i, j)) / samples);
        }
    }


}