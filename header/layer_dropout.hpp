#ifndef LAYER_DROPOUT_H
#define LAYER_DROPOUT_H

#include "tensor_inline.hpp"

class Layer_Dropout {

    public:
        
        Layer_Dropout(const double rate);

        const TensorInline& getOutput() const { return this->output; }
        const TensorInline& getDinputs() const{ return this->dinputs; }

        void forward(const TensorInline &inputs);
        void backward(const TensorInline &dvalues);


    private:
        TensorInline inputs, output;
        TensorInline dinputs, binary_mask;
        double rate;

};

#endif