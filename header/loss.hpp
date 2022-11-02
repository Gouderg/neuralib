#ifndef LOSS_H
#define LOSS_H

#include <cmath>

#include "../header/tensor.hpp"

class Loss {

    public:

        virtual ~Loss(){}

        // Function for inheritance.
        virtual std::vector<double> forward(Tensor& y_pred, Tensor& y_true);

        // Calculates the data and regularization losses given model output and ground truth values.
        double calculate(Tensor& output, Tensor& y);

};


class Loss_CategoricalCrossEntropy : public Loss {

    public:
        std::vector<double> forward(Tensor& y_pred, Tensor& y_true);

};



#endif