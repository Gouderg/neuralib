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

        Tensor& getWeightMomentum() { return this->weight_momentum; }
        Tensor& getBiasMomentum() { return this->bias_momentum; }
        Tensor& getWeightCache() { return this->weight_cache; }
        Tensor& getBiasCache() { return this->bias_cache; }

        Tensor getSquaredDWeights() { return this->dweights * this->dweights; }
        Tensor getSquaredDBias() { return this->dbiases * this->dbiases; }


        // Setter.
        void setWeights (std::vector<std::vector<double>> tensor) { this->weights.setTensor(tensor); }
        void setWeightMomentum (Tensor w) {this->weight_momentum = w;}
        void setBiasMomentum (Tensor b) {this->bias_momentum = b;}
        void setWeightCache (Tensor w) {this->weight_cache = w;}
        void setBiasCache (Tensor b) {this->bias_cache = b;}

        // Add.
        void addWeights (Tensor& t) {this->weights += t;}
        void addBiases (Tensor& t) {this->biases += t;}
        void addWeightMomentum (Tensor& w) {this->weight_momentum += w;}
        void addBiasMomentum (Tensor& b) {this->bias_momentum += b;}
        void addWeightCache (Tensor& w) {this->weight_cache += w;}
        void addBiasCache (Tensor& b) {this->bias_cache += b;}


        // Forward pass.
        void forward(Tensor& inputs);

        // Backward pass.
        void backward(Tensor &dvalues);


    private:
        Tensor inputs, weights, biases, output;
        Tensor dinputs, dweights, dbiases;
        Tensor weight_momentum, bias_momentum, weight_cache, bias_cache;

};

#endif