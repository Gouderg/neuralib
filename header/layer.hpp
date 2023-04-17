#ifndef LAYER_H
#define LAYER_H

#include "tensor_inline.hpp"

class Layer {

    public:
        virtual ~Layer(){};
        virtual void forward(const TensorInline& inputs, const bool training) = 0;
        virtual void backward(const TensorInline& dvalues) = 0;

        // Getter.
        const TensorInline& getOutput() const { return this->output; }
        const TensorInline& getDinputs() const{ return this->dinputs; }

        virtual bool isTrainable() = 0;

    protected:
        TensorInline output, dinputs;

};

class Layer_Input : public Layer {

    public:
        void forward(const TensorInline& inputs, const bool training = false);

        void backward(const TensorInline& dvalues) {};

        bool isTrainable() { return false; }
}; 

#endif