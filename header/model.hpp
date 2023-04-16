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


struct ModelParameters {
    TensorInline X, y;
    std::tuple<TensorInline, TensorInline> validatation_data;
    const int epochs = 1;
    const int print_every = 1;
};

class Model {

    public:
        Model();
        ~Model();

        void add(Layer* layer);
        void set(Loss* loss, Optimizer* opti, Accuracy* accuracy);
        void train(ModelParameters params);
        
        TensorInline forward(const TensorInline& X, const bool training);
        void backward(const TensorInline& output, const TensorInline& y);

        void test(int &a) { a += 1;}

    private:
        std::vector<Layer*> layers;
        Loss* loss;
        Optimizer* optimizer;
        Accuracy* accuracy;
};

#endif