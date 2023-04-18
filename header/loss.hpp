#ifndef LOSS_H
#define LOSS_H

#include <cmath>

#include "../header/layer_dense.hpp"

class Loss {

    public:

        // Destructor.
        virtual ~Loss(){}

        // Function for inheritance.
        virtual std::vector<double> forward(const TensorInline &y_pred, const TensorInline& y_true) = 0;

        virtual void backward(const TensorInline &dvalues, const TensorInline &y_true) = 0;

        // Calculates the data and regularization losses given model output and ground truth values.
        double calculate(const TensorInline& output, const TensorInline& y);

        double regularization_loss(const Layer_Dense& layer);

        // Calculate the accuracy
        static double accuracy(const TensorInline &inputs, const TensorInline &y);

        const TensorInline& getDinputs() const { return this->dinputs; }
    
    protected:
        TensorInline dinputs;


};


class Loss_CategoricalCrossEntropy : public Loss {

    public:
        std::vector<double> forward(const TensorInline &y_pred, const TensorInline& y_true);
    
        void backward(const TensorInline &dvalues, const TensorInline &y_true);
}; 

class Loss_BinaryCrossentropy : public Loss {

    public:
        std::vector<double> forward(const TensorInline &y_pred, const TensorInline& y_true);
    
        void backward(const TensorInline &dvalues, const TensorInline &y_true);
};

class Loss_MeanSquaredError : public Loss {

    public:
        std::vector<double> forward(const TensorInline &y_pred, const TensorInline& y_true);
    
        void backward(const TensorInline &dvalues, const TensorInline &y_true);

};

class Loss_MeanAbsoluteError : public Loss {
    
    public:
        std::vector<double> forward(const TensorInline &y_pred, const TensorInline& y_true);
    
        void backward(const TensorInline &dvalues, const TensorInline &y_true);
};

#endif