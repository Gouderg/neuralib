#ifndef LOSS_H
#define LOSS_H

#include <cmath>

#include "../header/layer_dense.hpp"

class Loss {

    public:

        virtual ~Loss(){}

        // Getter.

        // Function for inheritance.
        virtual std::vector<double> forward(TensorInline y_pred, TensorInline& y_true);

        // Calculates the data and regularization losses given model output and ground truth values.
        double calculate(TensorInline& output, TensorInline& y);

        double regularization_loss(Layer_Dense& layer);;

        // Calculate the accuracy
        static double accuracy(TensorInline &inputs, TensorInline y);


};


class Loss_CategoricalCrossEntropy : public Loss {

    public:
        std::vector<double> forward(TensorInline y_pred, TensorInline& y_true);
    
        void backward(TensorInline &dvalues, TensorInline &y_true);

        TensorInline& getDinputs() { return this->dinputs; }
        
    private:
        TensorInline dinputs;
}; 

#endif