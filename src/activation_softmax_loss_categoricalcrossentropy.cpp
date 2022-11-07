#include "../header/activation_softmax_loss_categoricalcrossentropy.hpp"

Activation_Softmax_Loss_CategoricalCrossentropy::Activation_Softmax_Loss_CategoricalCrossentropy() {
    this->activation = Activation_Softmax();
    this->loss = Loss_CategoricalCrossEntropy();
}

double Activation_Softmax_Loss_CategoricalCrossentropy::forward(Tensor &inputs, Tensor &y_true) {

    this->activation.forward(inputs);

    this->output = this->activation.getOutput();

    return this->loss.calculate(this->activation.getOutput(), y_true);
}

void Activation_Softmax_Loss_CategoricalCrossentropy::backward(Tensor &dvalues, Tensor& y_true) {

    this->dinputs = dvalues;

    // if labels are one-hot encoded.
    std::vector<double> y_flat;
    if (y_true.shapeY() == 2) {
        for (auto &row : y_true.getTensor()) {
            y_flat.push_back(std::max_element(row.begin(),row.end()) - row.begin());
        }
    } else {
        y_flat = y_true.getRow(0);
    }

    // Compute the gradient (-1 on the good label).
    double samples = y_flat.size();
    for (int i = 0; i < samples; i++) {
        this->dinputs.setValue(i, y_flat[i], this->dinputs.getValue(i, y_flat[i]) - 1 / samples);
    }
}