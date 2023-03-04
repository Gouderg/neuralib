#include "../header/layer_dense.hpp"

// Constructor.
Layer_Dense::Layer_Dense(const int n_inputs, const int n_neurons, double weight_reg_L1, double weight_reg_L2, double bias_reg_L1, double bias_reg_L2) {
    
    // Init layers.
    this->weights = TensorInline({n_inputs, n_neurons, true});
    this->biases = TensorInline({1, n_neurons});

    // Init optimizer layers with 0 with the same shape.
    this->weight_momentum = TensorInline({n_inputs, n_neurons});
    this->bias_momentum = TensorInline({1, n_neurons});
    this->weight_cache = TensorInline({n_inputs, n_neurons});
    this->bias_cache = TensorInline({1, n_neurons});

    this->weight_reg_L1 = weight_reg_L1;
    this->weight_reg_L2 = weight_reg_L2;
    this->bias_reg_L1 = bias_reg_L1;
    this->bias_reg_L2 = bias_reg_L2;

}

void Layer_Dense::forward(const TensorInline& inputs) {
    
    this->inputs = inputs;

    this->output = TensorInline::dot(inputs, this->weights) + this->biases;
}

void Layer_Dense::backward(const TensorInline &dvalues) {
    
    // Gradients on parameters.
    this->dweights = TensorInline::dot(this->inputs.transposate(), dvalues);
    this->dbiases = TensorInline({1, dvalues.getWidth()});

    
    int cpt = -1;
    for (int i = 0; i < dvalues.getHeight() * dvalues.getWidth(); i++) {
        if (i % dvalues.getHeight() == 0) { cpt += 1; }
        this->dbiases.tensor[cpt] += dvalues.tensor[i];
    }

    // Regularization.
    if (this->weight_reg_L1 > 0) {
        TensorInline w = TensorInline({this->weights.getHeight(), this->weights.getWidth(), false, 1});
        for (int i = 0; i < w.getHeight() * w.getWidth(); i++) {
            if (this->weights.tensor[i] < 0) {
                w.tensor[i] = -1;
            }
        }
        this->dweights += (w * this->weight_reg_L1); 
    }

    if (this->weight_reg_L2 > 0) {
        this->dweights += (this->weights * 2 * this->weight_reg_L2);
    }

    if (this->bias_reg_L1 > 0) {
        TensorInline b = TensorInline({this->biases.getHeight(), this->biases.getWidth(), false, 1});
        for (int i = 0; i < b.getHeight() * b.getWidth(); i++) {
            if (this->biases.tensor[i] < 0) {
                b.tensor[i] = -1;
            }
        }
        this->dbiases += (b * this->bias_reg_L1); 
    }

    if (this->bias_reg_L2 > 0) {
        this->dbiases += (this->biases * 2 * this->bias_reg_L2);
    }

    
    // Gradients on values.
    this->dinputs = TensorInline::dot(dvalues, this->weights.transposate());
}
