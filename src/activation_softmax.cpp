#include "../header/activation_softmax.hpp"

void Activation_Softmax::forward(Tensor inputs) {
    
    // Get the tensor.
    std::vector<std::vector<double>> t = inputs.getTensor();
    
    // Init the output size.
    this->output = Tensor(t.size(), t[0].size());
    
    // Init exp values.
    std::vector<std::vector<double>> exp_t(t.size(), std::vector<double> (t[0].size(), 0));
    
    // Init somme of row exp.
    std::vector<double> somme_exp (t.size(), 0);

    // Get all the exp_values.
    for (int i = 0; i < t.size(); i++) {
        for (int j = 0; j < t[0].size(); j++) {
            exp_t[i][j] = exp(t[i][j]);
            somme_exp[i] += exp(t[i][j]);
        }
    }

    // Store the output values.
    for (int i = 0; i < t.size(); i++) {
        for (int j = 0; j < t[0].size(); j++) {
            this->output.addValue(i, j, exp_t[i][j] / somme_exp[i]);
        }
    }


}