#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <tuple>
#include <random>
#include <cmath>

#include "tensor_inline.hpp"
#include "tensor.hpp"


class Dataset {

    public:
        static std::tuple<TensorInline, TensorInline> spiral_data(const int samples, const int classes);
        static std::tuple<Tensor, Tensor> spiral_data2(const int samples, const int classes);

};

#endif