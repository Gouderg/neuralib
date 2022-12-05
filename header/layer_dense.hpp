#ifndef LAYER_DENSE_H
#define LAYER_DENSE_H

#include "../header/tensor.hpp"

class Layer_Dense {

    public:

        // Constructor.
        Layer_Dense(){};
        Layer_Dense(const int n_inputs, const int n_neurons);
        
        // Getter.
        Tensor& getWeights() { return this->weights; }
        Tensor& getBiases() { return this->biases; }
        Tensor& getOutput() { return this->output; }
        Tensor& getDinputs() { return this->dinputs; }
        Tensor& getDweights() { return this->dweights; }
        Tensor& getDbiases() { return this->dbiases; }

        // Setter.
        void setWeights (std::vector<std::vector<double>> tensor) { this->weights.setTensor(tensor); }

        // Add.
        void addWeights (Tensor& t) {this->weights += t;}
        void addBiases (Tensor& t) {this->biases += t;}


        // Forward pass.
        void forward(Tensor& inputs);

        // Backward pass.
        void backward(Tensor &dvalues);


    private:
        Tensor inputs, weights, biases, output;
        Tensor dinputs, dweights, dbiases;

};

#endif