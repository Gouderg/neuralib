#ifndef LAYER_DROPOUT_H
#define LAYER_DROPOUT_H

#include "../header/tensor_inline.hpp"
#include "../header/layer.hpp"

class Layer_Dropout : public Layer {

    public:
        // Constructor.
        Layer_Dropout(const double rate);

        // Forward.
        void forward(const TensorInline &inputs, const bool training = false);
        
        // Backward.
        void backward(const TensorInline &dvalues);

        bool isTrainable() { return false; }

    private:
        TensorInline inputs;
        TensorInline binary_mask;
        double rate;

};

#endif