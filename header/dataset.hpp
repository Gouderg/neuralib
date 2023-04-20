#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <random>
#include <cmath>

#include "tensor_inline.hpp"


struct Data {
    TensorInline X, y;
};

class Dataset {

    public:
        
        static Data spiral_data(const int samples, const int classes);
        static Data sine_data(const int samples);
};

#endif