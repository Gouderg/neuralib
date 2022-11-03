#ifndef LAYER_DENSE_H
#define LAYER_DENSE_H

#include "../header/tensor.hpp"

class Layer_Dense {

    public:

        // Constructor.
        Layer_Dense(){};
        Layer_Dense(const int n_inputs, const int n_neurons);
        
        // Getter.
        Tensor& getOutput() { return this->output; }

        // Forward pass.
        void forward(Tensor& inputs);

        // Backward pass.
        void backward(Tensor &dvalues);


    private:
        Tensor inputs, weights, biases, output;
        Tensor dinputs, dweights, dbiases;

};

#endif