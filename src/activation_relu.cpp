#include "../header/activation_relu.hpp"

void Activation_ReLU::forward(Tensor inputs) {

    std::vector<std::vector<double>> t = inputs.getTensor();

    this->output = Tensor(t.size(), t[0].size());
    
    for (int i = 0; i < t.size(); i++) {
        for (int j = 0; j < t[0].size(); j++) {
            this->output.addValue(i, j, (t[i][j] > 0) ? t[i][j] : 0);
        }
    }
}