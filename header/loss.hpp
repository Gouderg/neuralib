#ifndef LOSS_H
#define LOSS_H

#include <cmath>

#include "../header/tensor.hpp"

class Loss {

    public:

        virtual ~Loss(){}

        // Getter.

        // Function for inheritance.
        virtual std::vector<double> forward(Tensor& y_pred, Tensor& y_true);

        // Calculates the data and regularization losses given model output and ground truth values.
        double calculate(Tensor& output, Tensor& y);

        // Calculate the accuracy
        double accuracy(Tensor &inputs, Tensor y);


};


class Loss_CategoricalCrossEntropy : public Loss {

    public:
        std::vector<double> forward(Tensor& y_pred, Tensor& y_true);
    
        void backward(Tensor &dvalues, Tensor &y_true);

        Tensor& getDinputs() { return this->dinputs; }
        
    private:
        Tensor dinputs;
}; 

#endif