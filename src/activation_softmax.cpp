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
    this->dinputs = TensorInline({0, 0});

    for (int i = 0; i < height; i += width) {
        
        TensorInline test ({width, 1});
        for (int j = 0; j < width; j ++) {
            test.tensor[j] = this->output.tensor[i+j];
        }

        TensorInline test2 ({width, width});
        for (int j = 0; j < width; j ++) {
            test2.tensor[j * width + j] = test.tensor[j];
        }

        try {
            TensorInline jacobian_matrix({width, width});
            jacobian_matrix = test2 - TensorInline::dot(test, test.transposate());

            std::vector<double> out(dvalues.tensor.begin() + i, dvalues.tensor.begin() + i + width + 1);

            TensorInline res = TensorInline::dot(jacobian_matrix, out);
            
            this->dinputs.tensor.insert(this->dinputs.tensor.end(), res.tensor.begin(), res.tensor.end());      

        } catch (const std::invalid_argument & error) {
            std::cout << "Error: Activation_Softmax::backward - " << error.what() << std::endl;
            exit(1);
        }        
    }
}