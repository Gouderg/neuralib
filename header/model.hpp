#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <vector>

#include "../header/tensor_inline.hpp"
#include "../header/layer.hpp"
#include "../header/accuracy.hpp"
#include "../header/optimizer.hpp"
#include "../header/loss.hpp"
#include "../header/activation_sigmoid.hpp"
#include "../header/activation_relu.hpp"
#include "../header/activation_linear.hpp"
#include "../header/plot.hpp"
#include "../header/statistic.hpp"

struct Data {
    TensorInline X, y;
};

struct ModelParameters {
    Data data;
    Data validatation_data;
    const int epochs = 1;
    const int print_every = 1;
    const bool printStatistic = true;
    const bool plotData = true;
};

class Model {

    public:
        Model();

        void add(Layer* layer);
        void set(Loss* loss, Optimizer* opti, Accuracy* accuracy);
        void train(ModelParameters p);
        
        TensorInline forward(const TensorInline& X, const bool training);
        void backward(const TensorInline& output, const TensorInline& y);

    private:
        std::vector<Layer*> layers;
        std::vector<Layer_Dense*> trainable_layers;
        Loss* loss;
        Optimizer* optimizer;
        Accuracy* accuracy;

        Statistic* stat;
        Plot* plt;
};

#endif