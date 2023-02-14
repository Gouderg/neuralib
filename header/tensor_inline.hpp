#ifndef TENSOR_INLINE_H 
#define TENSOR_ININE_H

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <functional>
#include <ctime>
#include <cmath>
#include <omp.h>
#include <sys/sysinfo.h>

#include "tensor.hpp"

const int nb_procs = std::max(omp_get_num_procs() - 1, 2);

class TensorInline {

    public:

        // Set Tensor attribut public to simplify operation.
        std::vector<double> tensor;
        
        // Constructor.
        TensorInline(){};
        TensorInline(const int nb_col, const int nb_row, const int whichInit);

        // Getter.
        int getWidth() const { return this->width; }
        int getHeight() const { return this->height; }
        std::vector<double> getTensor() const { return this->tensor; }

        // Setter.
        void setWidth(const int w) { this->width = w; }
        void setHeight(const int h) { this->height = h; }
        void setTensor(const std::vector<double> t) { this->tensor = t; }  

        TensorInline dot(TensorInline const &t2);
        static TensorInline dot(const TensorInline& t1, const TensorInline& t2);

        // Cout.
        friend std::ostream& operator <<(std::ostream&, const TensorInline&);


    private:
        int width, height;              // Width is number of parameters in one sample and height is the number of samples.
};


#endif