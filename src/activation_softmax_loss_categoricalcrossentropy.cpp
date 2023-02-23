#include "../header/activation_softmax_loss_categoricalcrossentropy.hpp"

Activation_Softmax_Loss_CategoricalCrossentropy::Activation_Softmax_Loss_CategoricalCrossentropy() {
    this->activation = Activation_Softmax();
    this->loss = Loss_CategoricalCrossEntropy();
}

double Activation_Softmax_Loss_CategoricalCrossentropy::forward(TensorInline &inputs, TensorInline &y_true) {

    this->activation.forward(inputs);

    this->output = this->activation.getOutput();

    return this->loss.calculate(this->activation.getOutput(), y_true);
}

void Activation_Softmax_Loss_CategoricalCrossentropy::backward(TensorInline &dvalues, TensorInline& y_true) {

    this->dinputs = dvalues;

    // if labels are one-hot encoded.
    std::vector<double> y_flat;
    if (y_true.getHeight() == 2) {
        for (int i = 0; i < y_true.getHeight() * y_true.getWidth(); i += y_true.getWidth()) {
            y_flat.push_back(std::max_element(y_true.tensor.begin() + i ,y_true.tensor.end() + i + y_true.getWidth()) - (y_true.tensor.begin() + i));
        }
    } else {
        y_flat = y_true.tensor;
    }

    // Compute the gradient (-1 on the good label).
    double samples = y_flat.size();
    for (int i = 0; i < samples; i++) {
        this->dinputs.tensor[i * this->dinputs.getWidth() + y_flat[i]] -= 1;
    }
    this->dinputs /= samples;
}