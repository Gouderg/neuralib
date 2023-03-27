#ifndef ACCURACY_H
#define ACCURACY_H

#include "tensor_inline.hpp"

class Accuracy {

    public:
        virtual double calculate(const TensorInline& predictions, const TensorInline& y) = 0;
        
        virtual void init(const TensorInline& y) = 0;

        virtual ~Accuracy(){};
};


class Accuracy_Categorical : public Accuracy {

    public:
        Accuracy_Categorical(const bool binary_init): binary(binary_init) {}

        double calculate(const TensorInline& predictions, const TensorInline& y);

        void init(const TensorInline& y){}


    private:
        bool binary;
};

class Accuracy_Regression : public Accuracy {
    
    public: 
        Accuracy_Regression(const double precision_init): precision(precision_init) {}

        void init(const TensorInline& y) {};
        void init(const TensorInline& y, const bool reinit = false);

        double calculate(const TensorInline& predictions, const TensorInline& y);
    
    private:
        double precision;
};
#endif