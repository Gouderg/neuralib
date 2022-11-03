#include "../header/activation_relu.hpp"

void Activation_ReLU::forward(Tensor& inputs) {
    this->inputs = inputs;

    this->output = Tensor(inputs.shapeY(), inputs.shapeX());
    
    for (int i = 0; i < inputs.shapeY(); i++) {
        for (int j = 0; j < inputs.shapeX(); j++) {
            this->output.setValue(i, j, (inputs.getValue(i,j) > 0) ? inputs.getValue(i,j) : 0);
        }
    }
}

void Activation_ReLU::backward(Tensor &dvalues) {
    this->dinputs = dvalues;

    for (int i = 0; i < this->dinputs.shapeY(); i++) {
        for (int j = 0; j < this->dinputs.shapeX(); j++) {
            if (this->inputs.getValue(i, j) <= 0) {
                this->dinputs.setValue(i, j, 0);
            }
        }
    }
}