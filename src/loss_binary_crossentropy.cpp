#include "../header/loss.hpp"

std::vector<double> Loss_BinaryCrossentropy::forward(const TensorInline &y_pred, const TensorInline& y_true) {

    // Clip data to prevent division by 0.
    TensorInline y_pred_clipped = TensorInline::clip(y_pred, 1e-7, 1 - 1e-7);

    TensorInline samples_output ({y_pred.getHeight(), y_pred.getWidth()});

    for (int i = 0; i < samples_output.getHeight() * samples_output.getWidth(); i++) { 
        samples_output.tensor[i] = -1 * (y_true.tensor[i] * log(y_pred_clipped.tensor[i]) + (1.0 - y_true.tensor[i]) * log(1.0 - y_pred_clipped.tensor[i]));
    }

    std::vector<double> correct_output (samples_output.getHeight(), 0.0);
    int w = samples_output.getWidth();

    for (int i = 0; i < samples_output.getHeight() * w; i += w) {        
        for (int j = 0; j < w; j++) {
            correct_output[i / w] += samples_output.tensor[i + j];
        }
        correct_output[i / w] /= w;
    }

    return correct_output;
}

void Loss_BinaryCrossentropy::backward(const TensorInline &dvalues, const TensorInline &y_true) {

    double samples = dvalues.getHeight();
    double outputs = dvalues.getWidth();

    TensorInline clipped_dvalues = TensorInline::clip(dvalues, 1e-7, 1 - 1e-7);
    this->dinputs = dvalues;

    // Calculate gradient.
    for (int i = 0; i < clipped_dvalues.getHeight() * clipped_dvalues.getWidth(); i++) {        
        this->dinputs.tensor[i] =  -(y_true.tensor[i] / clipped_dvalues.tensor[i] - (1.0 - y_true.tensor[i]) / (1.0 - clipped_dvalues.tensor[i])) / outputs;
    }

    // Normalize gradient.
    this->dinputs /= samples;
}