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
#include <cassert>

#include "constants.hpp"

// Number of processors allowed.
const int nb_procs = std::max(omp_get_num_procs() - 1, 2);

struct TensorInlineParameters {
    int height;
    int width;
    bool isRandom = false;
    double valueToSet = 0.0;
    const double randomFactor = 0.01;
};


struct TensorInlineBinomialParameters {
    int trials;
    double rate;
    int height;
    int width;
};

class TensorInline {

    public:

        // Set Tensor attribut public to simplify operation.
        std::vector<double> tensor;
        
        // Constructor.
        TensorInline(){};
        TensorInline(TensorInlineParameters p);
        TensorInline(const TensorInline &p);
        

        // Getter.
        const int getWidth() const noexcept { return this->width; }
        const int getHeight() const noexcept { return this->height; }
        std::vector<double> getTensor() const noexcept { return this->tensor; }

        // Setter.
        void setWidth(const int w) noexcept { this->width = w; }
        void setHeight(const int h) noexcept { this->height = h; }
        void setTensor(const std::vector<double> t) noexcept { this->tensor = t; }  

        // Reshape.
        void reshape(const int new_height, const int new_width);
        std::string shape() const ;

        /** Basics operations */

        // Addition.
        TensorInline operator + (TensorInline const &t2) const;
        TensorInline operator + (double const &n) const;
        friend TensorInline operator + (const double &n, TensorInline const &t2);
        void operator += (TensorInline const &t2);
        void operator += (double const &n);

        // Substraction.
        TensorInline operator - (TensorInline const &t2) const;
        TensorInline operator - (double const &n) const;
        friend TensorInline operator - (const double &n, TensorInline const &t2);
        void operator -= (TensorInline const &t2);
        void operator -= (double const &n);

        // Multiplication.
        TensorInline operator * (TensorInline const &t2) const;
        TensorInline operator * (double const &n) const;
        friend TensorInline operator * (const double &n, TensorInline const &t2);
        void operator *= (TensorInline const &t2);
        void operator *= (double const &n);

        // Division.
        TensorInline operator / (TensorInline const &t2) const;
        TensorInline operator / (double const &n) const;
        friend TensorInline operator / (const double &n, TensorInline const &t2);
        void operator /= (TensorInline const &t2);
        void operator /= (double const &n);

        // Comparaison.
        bool operator == (TensorInline const &t2) const;
        bool operator != (TensorInline const &t2) const;
        bool operator <= (const double &n) const;


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
        static double sum(std::vector<double> const &t);

        // Exponential of all term.
        static TensorInline exp(const TensorInline & t1);

        // Clipped the value between the range.
        static TensorInline clip(const TensorInline & t1, const double range_min, const double range_max);

        // Return the sign of the value.
        static int sign(const double n);

        // Return the standard deviation of a vector.
        static double standard_deviation(const TensorInline & t1);

        // Binomial distribution.
        static TensorInline binomial(const TensorInlineBinomialParameters p);

        // Mean.
        static double mean(const TensorInline & t);

        // Round to the nearest integer.
        static int round(const double n);

        // Cout.
        friend std::ostream& operator <<(std::ostream&, const TensorInline&);

        // Slice tensor.
        TensorInline slice(const TensorInline& t, const int step, const int batchSize);

        // Shuffle tensor inplace and apply on labels.
        void shuffle(TensorInline& X, TensorInline& y);


    private:
        int width, height;              // Width is number of parameters in one sample and height is the number of samples.
};


#endif