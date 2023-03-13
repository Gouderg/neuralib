#include "../header/loss.hpp"

std::vector<double> Loss_MeanSquaredError::forward(const TensorInline &y_pred, const TensorInline& y_true) {
    
    assert(y_pred.getHeight() == y_true.getHeight() && y_pred.getWidth() == y_true.getWidth());  // Ensure tensor are the same.

    std::vector<double> samples_ouput (y_pred.getHeight(), 0.0);

    int w = y_pred.getWidth();
    for (int i = 0; i < y_pred.getHeight(); i ++) {
        for (int j = 0; j < y_pred.getWidth(); j++) {
            samples_ouput[i] += std::pow(y_true.tensor[i * w + j] - y_pred.tensor[i * w + j] , 2);
        }
        samples_ouput[i] /= w;
    }

    return samples_ouput;
}

void Loss_MeanSquaredError::backward(const TensorInline &dvalues, const TensorInline &y_true) {
    this->dinputs = dvalues;
    double samples = dvalues.getHeight();
    double outputs = dvalues.getWidth();

    for (int i = 0; i < samples * outputs; i++ ) {
        this->dinputs.tensor[i] = -2.0 * (y_true.tensor[i] - dvalues.tensor[i]) / outputs;
    }

    this->dinputs /= samples;
}   

double Loss_MeanSquaredError::accuracy(const TensorInline &output, const TensorInline &y_true, const double accuracy_precision) {
    
    TensorInline predictions = output;
    double accuracy = 0.0;


    for (int i = 0; i < output.getHeight() * output.getWidth(); i++) {
        accuracy += (std::abs(output.tensor[i] - y_true.tensor[i]) < accuracy_precision) ? 1.0 : 0.0;
    }

    return accuracy / (predictions.getHeight() * predictions.getWidth());

}