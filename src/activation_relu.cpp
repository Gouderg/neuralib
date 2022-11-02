#include "../header/activation_relu.hpp"

void Activation_ReLU::forward(Tensor& inputs) {

    this->output = Tensor(inputs.shapeY(), inputs.shapeX());
    
    for (int i = 0; i < inputs.shapeY(); i++) {
        for (int j = 0; j < inputs.shapeX(); j++) {
            this->output.addValue(i, j, (inputs.getValue(i,j) > 0) ? inputs.getValue(i,j) : 0);
        }
    }
}