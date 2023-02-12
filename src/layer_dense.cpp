#include "../header/layer_dense.hpp"

// Constructor.
Layer_Dense::Layer_Dense(const int n_inputs, const int n_neurons, double weight_reg_L1, double weight_reg_L2, double bias_reg_L1, double bias_reg_L2) {
    
    // Init layers.
    this->weights = Tensor(n_inputs, n_neurons, 1);
    this->biases = Tensor(1, n_neurons);

    // Init optimizer layers with 0 with the same shape.
    this->weight_momentum = Tensor(n_inputs, n_neurons, 0);
    this->bias_momentum = Tensor(1, n_neurons, 0);
    this->weight_cache = Tensor(n_inputs, n_neurons, 0);
    this->bias_cache = Tensor(1, n_neurons, 0);

    this->weight_reg_L1 = weight_reg_L1;
    this->weight_reg_L2 = weight_reg_L2;
    this->bias_reg_L1 = bias_reg_L1;
    this->bias_reg_L2 = bias_reg_L2;

}

void Layer_Dense::forward(Tensor& inputs) {
    
    this->inputs = inputs;

    this->output = inputs.dot(this->weights) + this->biases;
}

void Layer_Dense::backward(Tensor &dvalues) {
    
    // Gradients on parameters.
    this->dweights = this->inputs.transposate().dot(dvalues);
    
    std::vector<double> somme(dvalues.shapeX(), 0);
    
    for (int i = 0; i < dvalues.shapeY(); i++) {
        for(int j = 0; j < dvalues.shapeX(); j++) {
            somme[j] += dvalues.getValue(i, j);
        }
    }

    this->dbiases = Tensor(1,dvalues.shapeX());
    this->dbiases.setRow(0, somme); 

    // Regularization.
    if (this->weight_reg_L1 > 0) {
        Tensor w = Tensor(this->weights.shapeY(), this->weights.shapeX(), 0);
        for (int i = 0; i < w.shapeY(); i++) {
            for (int j = 0; j < w.shapeX(); j++) {
                if (this->weights.getValue(i, j) < 0) {
                    w.setValue(i, j, -1);
                }
            }
        }
        this->dweights += (w * this->weight_reg_L1); 
    }

    if (this->weight_reg_L2 > 0) {
        this->dweights += (this->weights * 2 * this->weight_reg_L2);
    }

    if (this->bias_reg_L1 > 0) {
        Tensor b = Tensor(this->biases.shapeY(), this->biases.shapeX(), 0);
        for (int i = 0; i < b.shapeY(); i++) {
            for (int j = 0; j < b.shapeX(); j++) {
                if (this->biases.getValue(i, j) < 0) {
                    b.setValue(i, j, -1);
                }
            }
        }
        this->dbiases += (b * this->bias_reg_L1); 
    }

    if (this->bias_reg_L2 > 0) {
        this->dbiases += (this->biases * 2 * this->bias_reg_L2);
    }


    // Gradients on values.
    this->dinputs = dvalues.dot(this->weights.transposate());
}
