#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>

#include "layer_dense.hpp"

// We assume that momemtum and epsilon is a different variable but can be mutualize between all the class.
class Optimizer {

    public: 
        Optimizer(const double learning_rate = 1.0, const double decay = 0.0, const double mom_ep = 0.0);

        // Getter.
        double getLr() const { return this->learning_rate; }
        double getCurrentLr() const { return this->current_lr; }

        
        // Update.
        void pre_update_params();
        virtual void update_params(Layer_Dense &layer) = 0;
        void post_update_params();

        // Destructor.
        virtual ~Optimizer(){};

        // Cout.
        friend std::ostream& operator <<(std::ostream&, const Optimizer&);

    protected:
        double learning_rate, current_lr, decay, mom_ep;
        int iterations;

};

class Optimizer_SGD : public Optimizer {

    public:
        // Constructor.
        Optimizer_SGD(const double learning_rate = 1.0, const double decay = 0.0, const double mom_ep = 0.0) : Optimizer(learning_rate, decay, mom_ep) {}

        // Update.
        void update_params(Layer_Dense &layer);

        // Cout.
        friend std::ostream& operator <<(std::ostream&, const Optimizer_SGD&);
};

class Optimizer_Adagrad : public Optimizer {

    public:
        // Constructor.
        Optimizer_Adagrad(const double learning_rate = 1.0, const double decay = 0.0, const double mom_ep = 1e-7) : 
        Optimizer(learning_rate, decay, mom_ep) {}

        // Update.
        void update_params(Layer_Dense &layer);

        // Cout.
        friend std::ostream& operator <<(std::ostream&, const Optimizer_Adagrad&);
};

class Optimizer_RMSprop : public Optimizer {

    public:
        // Constructor.
        Optimizer_RMSprop(const double learning_rate = 0.001, const double decay = 0.0, const double mom_ep = 1e-7, const double rho = 0.9 ) : 
        Optimizer(learning_rate, decay, mom_ep), rho(rho) {}

        // Update.
        void update_params(Layer_Dense &layer);

        // Cout.
        friend std::ostream& operator <<(std::ostream&, const Optimizer_RMSprop&);

    private:
        double rho;
};

class Optimizer_Adam : public Optimizer {

    public:
        // Constructor.
        Optimizer_Adam(const double learning_rate = 0.001, const double decay = 0.0, const double mom_ep = 1e-7, const double beta1 = 0.9, const double beta2 = 0.999) : 
        Optimizer(learning_rate, decay, mom_ep), beta1(beta1), beta2(beta2) {}

        // Update.
        void update_params(Layer_Dense &layer);

        // Cout.
        friend std::ostream& operator <<(std::ostream&, const Optimizer_Adam&);


    private:
        double beta1, beta2;

};


#endif