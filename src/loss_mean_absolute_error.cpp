#include "../header/loss.hpp"

std::vector<double> Loss_MeanAbsoluteError::forward(const TensorInline &y_pred, const TensorInline& y_true) {
    assert(y_pred.getHeight() == y_true.getHeight() && y_pred.getWidth() == y_true.getWidth());  // Ensure tensor are the same.

    std::vector<double> samples_ouput (y_pred.getHeight(), 0.0);

    int w = y_pred.getWidth();
    for (int i = 0; i < y_pred.getHeight(); i ++) {
        for (int j = 0; j < y_pred.getWidth(); j++) {
            samples_ouput[i] += std::abs(y_true.tensor[i * w + j] - y_pred.tensor[i * w + j]);
        }
        samples_ouput[i] /= w;
    }

    return samples_ouput;
}

void Loss_MeanAbsoluteError::backward(const TensorInline &dvalues, const TensorInline &y_true) {
    this->dinputs = dvalues;
    double samples = dvalues.getHeight();
    double outputs = dvalues.getWidth();

    for (int i = 0; i < samples * outputs; i++ ) {
        this->dinputs.tensor[i] = TensorInline::sign(y_true.tensor[i] - dvalues.tensor[i])  / outputs;
    }

    this->dinputs /= samples;
}