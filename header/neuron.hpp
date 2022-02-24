#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <ctime>

class Neuron {
    public:
        
        // Constructeur.
        Neuron(int size);
        Neuron(double* inputs, double* weight, double bias);

        // Agrégation.
        double agregation();

        // Getter.
        std::vector<double> getInputs() const {return this->inputs;}
        // Setter.


    private:
        std::vector<double> inputs;   // Inputs n-size.
        std::vector<double> weight;   // weight n-size.
        double biais;    // biais.

};