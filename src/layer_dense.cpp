#include "../header/layer_dense.hpp"

// Constructor.
Layer_Dense::Layer_Dense(LayerDenseParameters p) {
    
    // Init layers.
    this->weights = TensorInline({p.n_inputs, p.n_neurons, true});
    this->biases = TensorInline({1, p.n_neurons});

    // Init optimizer layers with 0 with the same shape.
    this->weight_momentum = TensorInline({p.n_inputs, p.n_neurons});
    this->bias_momentum = TensorInline({1, p.n_neurons});
    this->weight_cache = TensorInline({p.n_inputs, p.n_neurons});
    this->bias_cache = TensorInline({1, p.n_neurons});

    this->weight_reg_L1 = p.weight_reg_L1;
    this->weight_reg_L2 = p.weight_reg_L2;
    this->bias_reg_L1 = p.bias_reg_L1;
    this->bias_reg_L2 = p.bias_reg_L2;

}

void Layer_Dense::forward(const TensorInline& inputs, const bool training) {
    
    this->inputs = inputs;
    this->output = TensorInline::dot(inputs, this->weights) + this->biases;
}

void Layer_Dense::backward(const TensorInline &dvalues) {
    
    // Gradients on parameters.
    this->dweights = TensorInline::dot(this->inputs.transposate(), dvalues);
    this->dbiases = TensorInline({1, dvalues.getWidth()});
    for (int i = 0; i < dvalues.getHeight() * dvalues.getWidth(); i += dvalues.getWidth()) {
        for (int j = 0; j < dvalues.getWidth(); j++) {
            this->dbiases.tensor[j] += dvalues.tensor[i + j];
        }
    }

    // Regularization.
    if (this->weight_reg_L1 > 0) {
        TensorInline w = TensorInline({this->weights.getHeight(), this->weights.getWidth(), false, 1.0});
        for (int i = 0; i < w.getHeight() * w.getWidth(); i++) {
            if (this->weights.tensor[i] < 0) {
                w.tensor[i] = -1.0;
            }
        }
        this->dweights += (w * this->weight_reg_L1); 
    }

    if (this->weight_reg_L2 > 0) {
        this->dweights += (this->weights * 2.0 * this->weight_reg_L2);
    }

    if (this->bias_reg_L1 > 0) {
        TensorInline b = TensorInline({this->biases.getHeight(), this->biases.getWidth(), false, 1.0});
        for (int i = 0; i < b.getHeight() * b.getWidth(); i++) {
            if (this->biases.tensor[i] < 0) {
                b.tensor[i] = -1.0;
            }
        }
        this->dbiases += (b * this->bias_reg_L1); 
    }

    if (this->bias_reg_L2 > 0) {
        this->dbiases += (this->biases * 2.0 * this->bias_reg_L2);
    }

    // Gradients on values.
    this->dinputs = TensorInline::dot(dvalues, this->weights.transposate());
}
