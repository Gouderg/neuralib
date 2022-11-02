#include "../header/activation_softmax.hpp"

void Activation_Softmax::forward(Tensor& inputs) {
    
    int len_row = inputs.shapeX();
    int len_col = inputs.shapeY();

    // Init the output size.
    this->output = Tensor(len_col, len_row);
    
    // Init exp values.
    std::vector<std::vector<double>> exp_t(len_col, std::vector<double> (len_row, 0));
    
    // Init somme of row exp.
    std::vector<double> somme_exp (len_col, 0);

    // Get all the exp_values.
    for (int i = 0; i < len_col; i++) {
        for (int j = 0; j < len_row; j++) {
            exp_t[i][j] = exp(inputs.getValue(i, j));
            somme_exp[i] += exp(inputs.getValue(i, j));
        }
    }

    // Store the output values.
    for (int i = 0; i < len_col; i++) {
        for (int j = 0; j < len_row; j++) {
            this->output.addValue(i, j, exp_t[i][j] / somme_exp[i]);
        }
    }


}