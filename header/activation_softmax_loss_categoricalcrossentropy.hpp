#ifndef ACTIVATION_SOFT_LOSS_CROSS_H
#define ACTIVATION_SOFT_LOSS_CROSS_H

#include "activation_softmax.hpp"
#include "loss.hpp"

class Activation_Softmax_Loss_CategoricalCrossentropy {

    public:
        
        Activation_Softmax_Loss_CategoricalCrossentropy();

        // Forward pass.
        double forward(Tensor &ouput, Tensor &y_true);

        // Backward pass.
        void backward(Tensor &dvalues, Tensor &y_true);


    private:
        Activation_Softmax activation;
        Loss_CategoricalCrossEntropy loss;

        Tensor dinputs;

};


#endif