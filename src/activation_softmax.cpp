#include "../header/activation_softmax.hpp"

void Activation_Softmax::forward(Tensor& inputs) {

    this->inputs = inputs;
    
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
            this->output.setValue(i, j, exp_t[i][j] / somme_exp[i]);
        }
    }


}

void Activation_Softmax::backward(Tensor &dvalues) {
    this->dinputs = Tensor(dvalues.shapeY(), dvalues.shapeX());

    for (int i = 0; i < dvalues.shapeY(); i ++) {
        
        std::vector<std::vector<double>> test (dvalues.shapeX(), std::vector<double> (1, 0));
        for (int j = 0; j < dvalues.shapeX(); j ++) {
            test[j][0] = this->output.getValue(i, j);
        }

        Tensor test2 (dvalues.shapeX(), dvalues.shapeX());
        for (int j = 0; j < dvalues.shapeX(); j ++) {
            test2.setValue(j, j, test[j][0]);
        }

        Tensor jacobian_matrix(dvalues.shapeX(), dvalues.shapeX());
        jacobian_matrix = test2 - Tensor::dot(test, Tensor::transposate(test));

        std::vector<double> out = dvalues.getRow(i);

        this->dinputs.setRow(i, jacobian_matrix.dot(out));       
        
    }

}