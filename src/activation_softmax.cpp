#include "../header/activation_softmax.hpp"

void Activation_Softmax::forward(const TensorInline& inputs) {

    this->inputs = inputs;

    // Init the output size.
    this->output = TensorInline({inputs.getHeight(), inputs.getWidth()});
    
    // Init exp values.
    TensorInline exp_t({inputs.getHeight(), inputs.getWidth()});
    
    // Init somme of row exp.
    std::vector<double> somme_exp (inputs.getHeight(), 0);

    // Get all the exp_values.
    int row = -1;
    for (int i = 0; i < inputs.getHeight() * inputs.getWidth(); i++) {
        if (i % inputs.getWidth() == 0) {row += 1;}
        exp_t.tensor[i] = exp(inputs.tensor[i]);
        somme_exp[row] += exp(inputs.tensor[i]);
    }

    // Store the output values.
    row = -1;
    for (int i = 0; i < inputs.getHeight() * inputs.getWidth(); i++) {
        if (i % inputs.getWidth() == 0) {row += 1;}
        this->output.tensor[i] = exp_t.tensor[i] / somme_exp[row];
    }


}

void Activation_Softmax::backward(const TensorInline &dvalues) {

    int width = dvalues.getWidth();
    int height = dvalues.getHeight();
    this->dinputs = TensorInline({height, width});
    this->dinputs.tensor = {};

    for (int i = 0; i < height * width; i += width) {
        
        TensorInline single_dvalues ({width, 1});
        for (int j = 0; j < width; j ++) {
            single_dvalues.tensor[j] = dvalues.tensor[i+j];
        }

        TensorInline single_output_diag ({width, width});
        TensorInline single_output ({width, 1});
        for (int j = 0; j < width; j ++) {
            single_output_diag.tensor[j * width + j] = this->output.tensor[i+j];
            single_output.tensor[j] = this->output.tensor[i+j];
        }

        TensorInline jacobian_matrix({width, width});
        jacobian_matrix = single_output_diag - TensorInline::dot(single_output, single_output.transposate());

        TensorInline res = TensorInline::dot(jacobian_matrix, single_dvalues);

        this->dinputs.tensor.insert(this->dinputs.tensor.end(), res.tensor.begin(), res.tensor.end());
    }
}