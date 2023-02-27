#ifndef ACTIVATION_SOFT_LOSS_CROSS_H
#define ACTIVATION_SOFT_LOSS_CROSS_H

#include "activation_softmax.hpp"
#include "loss.hpp"

class Activation_Softmax_Loss_CategoricalCrossentropy {

    public:
        
        Activation_Softmax_Loss_CategoricalCrossentropy();

        // Forward pass.
        double forward(const TensorInline &ouput, const TensorInline &y_true);

        // Backward pass.
        void backward(const TensorInline &dvalues, const TensorInline &y_true);

        // Getter.
        const TensorInline& getOutput() const { return this->output; }
        const TensorInline& getDinputs() const { return this->dinputs; }
        Loss_CategoricalCrossEntropy& getLoss() {return this->loss;}



    private:
        Activation_Softmax activation;
        Loss_CategoricalCrossEntropy loss;

        TensorInline dinputs, output;

};


#endif