#ifndef ACTIVATION_RELU_H
#define ACTIVATION_RELU_H

#include "../header/tensor_inline.hpp"
#include "../header/layer.hpp"

class Activation_ReLU : public Layer {

    public:

        // Forward pass.
        void forward(const TensorInline &inputs, const bool training = false);

        // Backward pass.
        void backward(const TensorInline &dvalues);

    private:
        TensorInline inputs;
};

#endif