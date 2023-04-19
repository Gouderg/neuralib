#ifndef ACTIVATION_SOFT_LOSS_CROSS_H
#define ACTIVATION_SOFT_LOSS_CROSS_H

#include "activation_softmax.hpp"
#include "loss.hpp"

class Activation_Softmax_Loss_CategoricalCrossentropy {

    public:
        
        // Backward pass.
        static TensorInline backward(const TensorInline &dvalues, const TensorInline &y_true);

};


#endif