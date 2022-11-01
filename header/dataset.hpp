#pragma once

#include <iostream>
#include <tuple>
#include <random>
#include <cmath>

#include "tensor.hpp"

class Dataset {

    public:
        static std::tuple<Tensor, Tensor> spiral_data(const int samples, const int classes);
};