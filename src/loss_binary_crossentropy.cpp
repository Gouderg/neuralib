#include "../header/loss.hpp"

std::vector<double> Loss_BinaryCrossentropy::forward(const TensorInline &y_pred, const TensorInline& y_true) {

    // Clip data to prevent division by 0.
    TensorInline y_pred_clipped = y_pred;

    for (int i = 0; i < y_pred_clipped.getHeight() * y_pred_clipped.getWidth(); i++) {        
        if ((y_pred_clipped.tensor[i] < 1e-7)) {
            y_pred_clipped.tensor[i] = 1e-7;
        }

        if ((y_pred_clipped.tensor[i] > 1.0 - 1e-7)) {
            y_pred_clipped.tensor[i] = 1.0 - 1e-7;
        }
    }

    TensorInline samples_output ({y_pred.getHeight(), y_pred.getWidth()});

    for (int i = 0; i < samples_output.getHeight() * samples_output.getWidth(); i++) {        
        samples_output.tensor[i] =  -(y_true.tensor[i] * log(y_pred_clipped.tensor[i]) + (1.0 - y_true.tensor[i]) * log(1.0 - y_pred_clipped.tensor[i]));
    }

    std::vector<double> correct_output (samples_output.getHeight(), 0.0);
    for (int i = 0; i < samples_output.getHeight() * samples_output.getWidth(); i += samples_output.getWidth()) {        
        for (int j = 0; j < samples_output.getWidth(); j++) {
            correct_output[i] += samples_output.tensor[i + j];
        }
        correct_output[i] /= samples_output.getWidth();
    }

    return correct_output;
}

void Loss_BinaryCrossentropy::backward(const TensorInline &dvalues, const TensorInline &y_true) {

    double samples = dvalues.getHeight();
    double outputs = dvalues.getWidth();

    TensorInline clipped_dvalues = dvalues;
    this->dinputs = dvalues;


    for (int i = 0; i < clipped_dvalues.getHeight() * clipped_dvalues.getWidth(); i++) {        
        if ((clipped_dvalues.tensor[i] < 1e-7)) {
            clipped_dvalues.tensor[i] = 1e-7;
        }

        if ((clipped_dvalues.tensor[i] > 1.0 - 1e-7)) {
            clipped_dvalues.tensor[i] = 1.0 - 1e-7;
        }
    }

    for (int i = 0; i < clipped_dvalues.getHeight() * clipped_dvalues.getWidth(); i++) {        
        this->dinputs.tensor[i] =  -(y_true.tensor[i] / clipped_dvalues.tensor[i] - (1.0 - y_true.tensor[i]) / (1.0 - clipped_dvalues.tensor[i])) / outputs;
    }

    this->dinputs /= samples;
}