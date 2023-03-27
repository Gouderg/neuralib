#ifndef ACTIVATION_LINEAR_H
#define ACTIVATION_LINEAR_H

#include "../header/layer.hpp"
#include "../header/tensor_inline.hpp"

class Activation_Linear: public Layer {

    public:

        // Forward pass.
        void forward(const TensorInline &inputs, const bool training = false);

        // Backward pass.
        void backward(const TensorInline &dvalues);

    private:
        TensorInline inputs;
};

#endif