#include "../header/neuron.hpp"

Neuron::Neuron(int size) {
    for (int i = 0; i < size; i++) {
        this->inputs.push_back(rand() % 100 * 0.01);
        this->weight.push_back(rand() % 100 * 0.01);
    }
    this->biais = rand() % 10 * 0.1;
}

Neuron::Neuron(double* inputs, double* weight, double bias) {
    
}

double Neuron::agregation() {
    double agregation = 0;
    for (int i = 0; i < this->inputs.size(); i++) {
        agregation += this->inputs[i] * this->weight[i];
    }
    agregation += this->biais;
    return agregation;
}