#include "../header/loss.hpp"

std::vector<double> Loss_CategoricalCrossEntropy::forward(const TensorInline &y_pred, const TensorInline& y_true) {

    // Clip data to prevent division by 0.
    TensorInline y_pred_clipped = TensorInline::clipped(y_pred, 1e-15, 1 - 1e-15);

    // Probabilities for target values only if categoricals values.
    std::vector<double> correct_confidences(y_true.getWidth(), 0.0);

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

    double samples = dvalues.getHeight();
    double labels = dvalues.getWidth();
    this->dinputs = TensorInline({static_cast<int>(samples), static_cast<int>(labels)});

    TensorInline y_flat_diag({static_cast<int>(samples), static_cast<int>(labels)});
    if (y_true.getHeight() == 1) {
        for (int i = 0; i < samples; i ++) {
            y_flat_diag.tensor[i * y_flat_diag.getWidth() + y_true.tensor[i]] = 1.0;
        }
    } else {
        y_flat_diag = y_true;
    }

    for (int i = 0; i < samples * labels; i ++) {
        this->dinputs.tensor[i] = (-y_flat_diag.tensor[i] / dvalues.tensor[i] / samples);
    }
}