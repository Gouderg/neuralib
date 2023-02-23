#ifndef LAYER_DENSE_H
#define LAYER_DENSE_H

#include "../header/tensor_inline.hpp"

class Layer_Dense {

    public:

        // Constructor.
        Layer_Dense(){};
        Layer_Dense(const int n_inputs, const int n_neurons, double weight_reg_L1 = 0.0, double weight_reg_L2 = 0.0, double bias_reg_L1 = 0.0, double bias_reg_L2 = 0.0);
        
        // Getter.
        TensorInline& getWeights() { return this->weights; }
        TensorInline& getBiases() { return this->biases; }
        TensorInline& getOutput() { return this->output; }
        TensorInline& getDinputs() { return this->dinputs; }
        TensorInline& getDweights() { return this->dweights; }
        TensorInline& getDbiases() { return this->dbiases; }

        TensorInline& getWeightMomentum() { return this->weight_momentum; }
        TensorInline& getBiasMomentum() { return this->bias_momentum; }
        TensorInline& getWeightCache() { return this->weight_cache; }
        TensorInline& getBiasCache() { return this->bias_cache; }

        double getWeightRegL1() { return this->weight_reg_L1; }
        double getWeightRegL2() { return this->weight_reg_L2; }
        double getBiasRegL1() { return this->bias_reg_L1; }
        double getBiasRegL2() { return this->bias_reg_L2; }


        TensorInline getSquaredDWeights() { return this->dweights * this->dweights; }
        TensorInline getSquaredDBias() { return this->dbiases * this->dbiases; }


        // Setter.
        void setWeights (TensorInline w) { this->weights = w; }
        void setWeights (std::vector<double> w) { this->weights.tensor = w; }

        void setWeightMomentum (TensorInline w) {this->weight_momentum = w;}
        void setBiasMomentum (TensorInline b) {this->bias_momentum = b;}
        void setWeightCache (TensorInline w) {this->weight_cache = w;}
        void setBiasCache (TensorInline b) {this->bias_cache = b;}

        // Add.
        void addWeights (TensorInline& t) {this->weights += t;}
        void addBiases (TensorInline& t) {this->biases += t;}
        void addWeightMomentum (TensorInline& w) {this->weight_momentum += w;}
        void addBiasMomentum (TensorInline& b) {this->bias_momentum += b;}
        void addWeightCache (TensorInline& w) {this->weight_cache += w;}
        void addBiasCache (TensorInline& b) {this->bias_cache += b;}


        // Forward pass.
        void forward(TensorInline& inputs);

        // Backward pass.
        void backward(TensorInline &dvalues);


    private:
        TensorInline inputs, weights, biases, output;
        TensorInline dinputs, dweights, dbiases;
        TensorInline weight_momentum, bias_momentum, weight_cache, bias_cache;

        double weight_reg_L1, weight_reg_L2, bias_reg_L1, bias_reg_L2;
};

#endif