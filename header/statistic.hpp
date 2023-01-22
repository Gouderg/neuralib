#ifndef STATISTIC_H
#define STATISTIC_H

#include <iostream>
#include <vector>

#include "plot.hpp"

class Statistic {

    public:
        
        // Constructor.
        Statistic(){}

        // Add value.
        void update(const double loss, const double accuracy, const double learning_rate);

        // Plot graph.
        void plot();

    private:
        std::vector<double> loss;
        std::vector<double> accuracy;
        std::vector<double> learning_rate;

        Plot graph;

};


#endif