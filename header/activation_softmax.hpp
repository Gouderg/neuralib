#ifndef ACTIVATION_SOFTMAX_H
#define ACTIVATION_SOFTMAX_H

#include <cmath>

#include "../header/tensor_inline.hpp"
#include "../header/layer.hpp"

class Activation_Softmax : public Layer {
    
    public:

        // Forward pass.
        void forward(const TensorInline& inputs, const bool training = false);

        // Backward pass.
        void backward(const TensorInline &dvalues);

    private:
        TensorInline inputs;
};

#endif