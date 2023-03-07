#ifndef LAYER_DROPOUT_H
#define LAYER_DROPOUT_H

#include "tensor_inline.hpp"

class Layer_Dropout {

    public:
        // Constructor.
        Layer_Dropout(const double rate);

        // Getter.
        const TensorInline& getOutput() const { return this->output; }
        const TensorInline& getDinputs() const{ return this->dinputs; }

        // Forward.
        void forward(const TensorInline &inputs);
        
        // Backward.
        void backward(const TensorInline &dvalues);


    private:
        TensorInline inputs, output;
        TensorInline dinputs, binary_mask;
        double rate;

};

#endif