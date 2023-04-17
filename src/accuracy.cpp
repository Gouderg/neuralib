#include "../header/accuracy.hpp"

double Accuracy_Categorical::calculate(const TensorInline& predictions, const TensorInline& y) {
    
    // Get the indice of the best score.
    std::vector<int> pred;
    for (int i = 0; i < predictions.getHeight() * predictions.getWidth(); i += predictions.getWidth()) {
        pred.push_back(std::max_element(predictions.tensor.begin() + i , predictions.tensor.begin() + i + predictions.getWidth()) - (predictions.tensor.begin() + i));
    }

    // get the indice of the best ground truth.
    std::vector<double> y_flat;
    if (this->binary && y.getHeight() == 2 ) {
        for (int i = 0; i < y.getHeight() * y.getWidth(); i += y.getWidth()) {
            y_flat.push_back(std::max_element(y.tensor.begin() + i , y.tensor.begin() + i + y.getWidth()) - (y.tensor.begin() + i));
        }
    } else {
        y_flat = y.tensor;
    }

    // Compute the mean.
    double somme = 0.0;
    for (int i = 0; i < static_cast<int>(pred.size()); i++) {
        if (pred[i] == y_flat[i]) {
            somme += 1.0;
        }
    }

    return somme / pred.size();
}

double Accuracy_Regression::calculate(const TensorInline& predictions, const TensorInline& y) {

    double accuracy = 0.0;

    for (int i = 0; i < predictions.getHeight() * predictions.getWidth(); i++) {
        accuracy += (std::abs(predictions.tensor[i] - y.tensor[i]) < this->precision) ? 1.0 : 0.0;
    }
    
    return accuracy / (predictions.getHeight() * predictions.getWidth());
}

void Accuracy_Regression::init(const TensorInline& y, const bool reinit) {
    if (this->precision || reinit) {
        this->precision = TensorInline::standard_deviation(y) / STRICT_ACCURACY_METRICS;
    }
}