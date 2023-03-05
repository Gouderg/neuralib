#ifndef TENSOR_INLINE_H 
#define TENSOR_INLINE_H

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

#include "constants.hpp"

const int nb_procs = std::max(omp_get_num_procs() - 1, 2);

struct TensorInlineParams {
    int height;
    int width;
    bool isRandom = false;
    double valueToSet = 0.0;
};

class TensorInline {

    public:

        // Set Tensor attribut public to simplify operation.
        std::vector<double> tensor;
        
        // Constructor.
        TensorInline(){};
        TensorInline(TensorInlineParams p);

        // Getter.
        const int getWidth() const { return this->width; }
        const int getHeight() const { return this->height; }
        std::vector<double> getTensor() const { return this->tensor; }

        // Setter.
        void setWidth(const int w) { this->width = w; }
        void setHeight(const int h) { this->height = h; }
        void setTensor(const std::vector<double> t) { this->tensor = t; }  

        /** Basics operations */

        // Addition.
        TensorInline operator + (TensorInline const &t2) const;
        TensorInline operator + (double const &n) const;
        void operator += (TensorInline const &t2);
        void operator += (double const &n);

        // Substraction.
        TensorInline operator - (TensorInline const &t2) const;
        TensorInline operator - (double const &n) const;
        void operator -= (TensorInline const &t2);
        void operator -= (double const &n);

        // Multiplication.
        TensorInline operator * (TensorInline const &t2) const;
        TensorInline operator * (double const &n) const;
        void operator *= (TensorInline const &t2);
        void operator *= (double const &n);

        // Division.
        TensorInline operator / (TensorInline const &t2) const;
        TensorInline operator / (double const &n) const;
        void operator /= (TensorInline const &t2);
        void operator /= (double const &n);

        // Comparaison.
        bool operator == (TensorInline const &t2) const;
        bool operator != (TensorInline const &t2) const;


        /** Special operations */

        // Dot product. No need product from one tensor. Need two tensors to apply multi-procs.
        static TensorInline dot(const TensorInline& t1, const TensorInline& t2);
        static TensorInline dot(const TensorInline& t1, const std::vector<double>& t2);


        // Square root.
        TensorInline sqrt() const;

        // Absolute value.
        TensorInline abs() const;

        // Transposate.
        TensorInline transposate() const;

        // Sum all terms of the tensor.
        static double sum(TensorInline const &t);

        // Cout.
        friend std::ostream& operator <<(std::ostream&, const TensorInline&);


    private:
        int width, height;              // Width is number of parameters in one sample and height is the number of samples.
};


#endif