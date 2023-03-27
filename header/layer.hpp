#ifndef LAYER_H
#define LAYER_H

#include "tensor_inline.hpp"

class Layer {

    public:
        virtual ~Layer(){};
        virtual void forward(const TensorInline& inputs, const bool training) = 0;
        virtual void backward(const TensorInline& dvalues) = 0;

};

class Layer_Input : public Layer {

    public:
        void forward(const TensorInline& inputs, const bool training);

        void backward(const TensorInline& dvalues) {};

        const TensorInline& getOutput() const { return this->output; }

    private:
        TensorInline output;
}; 

#endif