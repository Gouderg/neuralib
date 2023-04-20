#ifndef LOSS_H
#define LOSS_H

#include <cmath>

#include "../header/layer_dense.hpp"


struct LossValues {
    double data_loss, regularization_loss;
};

class Loss {

    public:

        // Destructor.
        virtual ~Loss(){}

        // Function for inheritance.
        virtual std::vector<double> forward(const TensorInline &y_pred, const TensorInline& y_true) = 0;

        virtual void backward(const TensorInline &dvalues, const TensorInline &y_true) = 0;

        // Calculates the data and regularization losses given model output and ground truth values.
        LossValues calculate(const TensorInline& output, const TensorInline& y, const bool with_regularization = false);

        double regularization_loss();

        const TensorInline& getDinputs() const { return this->dinputs; }

        void setTrainableLayer(const std::vector<Layer_Dense*> layers) { this->trainable_layers = layers; }
    
    protected:
        TensorInline dinputs;
        std::vector<Layer_Dense*> trainable_layers;

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