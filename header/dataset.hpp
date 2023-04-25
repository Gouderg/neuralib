#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <fstream>
#include <random>
#include <cmath>

#include "tensor_inline.hpp"
#include "tools.hpp"


struct Data {
    TensorInline X, y;
};

enum FileType {
    labels = 2049,
    images = 2051
};

enum ScaleFormat {
    between0And1, betweenMinus1And1
};

class Dataset {

    public:
        
        static Data spiral_data(const int samples, const int classes);
        static Data sine_data(const int samples);
        static TensorInline read_idx_file(const std::string path, const FileType fileType);
        static void scale_pixels_values(TensorInline& X, ScaleFormat scale);
};

#endif