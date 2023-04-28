#ifndef ACCURACY_H
#define ACCURACY_H

#include "tensor_inline.hpp"

class Accuracy {

    public:

        Accuracy();

        virtual double calculate(const TensorInline& predictions, const TensorInline& y) = 0;
        
        virtual void init(const TensorInline& y, const bool reinit = false) = 0;

        virtual ~Accuracy(){};

        void new_pass();
        double calculate_accumulated();
    
    protected:
        double accumulated_sum, accumulated_count;
};


class Accuracy_Categorical : public Accuracy {

    public:
        Accuracy_Categorical(const bool binary_init): Accuracy(), binary(binary_init) {}

        double calculate(const TensorInline& predictions, const TensorInline& y);

        void init(const TensorInline& y, const bool reinit = false){}


    private:
        bool binary;
};

class Accuracy_Regression : public Accuracy {
    
    public: 
        Accuracy_Regression(const double precision_init): Accuracy(), precision(precision_init) {}

        void init(const TensorInline& y, const bool reinit = false);

        double calculate(const TensorInline& predictions, const TensorInline& y);
    
    private:
        double precision;
};

class Accuracy_Binary : public Accuracy {
    
    public: 

        Accuracy_Binary(): Accuracy() {}

        void init(const TensorInline& y, const bool reinit = false) {};

        double calculate(const TensorInline& predictions, const TensorInline& y);
};
#endif