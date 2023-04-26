#ifndef LAYER_DENSE_H
#define LAYER_DENSE_H

#include "../header/tensor_inline.hpp"
#include "../header/layer.hpp"

struct LayerDenseOptions {
    const int n_inputs;
    const int n_neurons;
    const double weight_reg_L1 = 0.0;
    const double weight_reg_L2 = 0.0;
    const double bias_reg_L1 = 0.0;
    const double bias_reg_L2 = 0.0;
    const double randomFactor = 0.01;
};

struct LayerDenseParameters {
    TensorInline weights, biases;
};

class Layer_Dense : public Layer {

    public:

        // Constructor.
        Layer_Dense(){};
        Layer_Dense(LayerDenseOptions p);
        
        // Getter.
        const TensorInline& getWeights() const { return this->weights; }
        const TensorInline& getBiases() const { return this->biases; }
        const TensorInline& getDweights() const { return this->dweights; }
        const TensorInline& getDbiases() const { return this->dbiases; }

        const TensorInline& getWeightMomentum() const { return this->weight_momentum; }
        const TensorInline& getBiasMomentum() const { return this->bias_momentum; }
        const TensorInline& getWeightCache() const { return this->weight_cache; }
        const TensorInline& getBiasCache() const { return this->bias_cache; }

        const double getWeightRegL1() const { return this->weight_reg_L1; }
        const double getWeightRegL2() const { return this->weight_reg_L2; }
        const double getBiasRegL1() const { return this->bias_reg_L1; }
        const double getBiasRegL2() const { return this->bias_reg_L2; }


        const TensorInline getSquaredDWeights() const { return this->dweights * this->dweights; }
        const TensorInline getSquaredDBias() const { return this->dbiases * this->dbiases; }


        // Setter.
        void setWeights (TensorInline w) { this->weights = w; }
        void setWeights (std::vector<double> w) { this->weights.tensor = w; }

        void setWeightMomentum (const TensorInline w) {this->weight_momentum = w;}
        void setBiasMomentum (const TensorInline b) {this->bias_momentum = b;}
        void setWeightCache (const TensorInline w) {this->weight_cache = w;}
        void setBiasCache (const TensorInline b) {this->bias_cache = b;}

        // Add.
        void addWeights (const TensorInline& t) {this->weights += t;}
        void addBiases (const TensorInline& t) {this->biases += t;}
        void addWeightMomentum (const TensorInline& w) {this->weight_momentum += w;}
        void addBiasMomentum (const TensorInline& b) {this->bias_momentum += b;}
        void addWeightCache (const TensorInline& w) {this->weight_cache += w;}
        void addBiasCache (const TensorInline& b) {this->bias_cache += b;}


        // Forward pass.
        void forward(const TensorInline& inputs, const bool training = false);

        // Backward pass.
        void backward(const TensorInline &dvalues);

        bool isTrainable() { return true; }

        // Returns the weight and bias.
        LayerDenseParameters getParameters() { return {.weights=this->weights, .biases=this->biases }; }

        // Set the weights and bias.
        void setParameters(LayerDenseParameters params);

    private:
        TensorInline inputs, weights, biases;
        TensorInline dweights, dbiases;
        TensorInline weight_momentum, bias_momentum, weight_cache, bias_cache;

        double weight_reg_L1, weight_reg_L2, bias_reg_L1, bias_reg_L2;
};

#endif