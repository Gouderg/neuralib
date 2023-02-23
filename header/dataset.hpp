#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <tuple>
#include <random>
#include <cmath>

#include "tensor_inline.hpp"


class Dataset {

    public:
        static std::tuple<TensorInline, TensorInline> spiral_data(const int samples, const int classes);

};

#endif