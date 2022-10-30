#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <functional>
#include <ctime>

const double MEAN = 0.0;
const double DEVIATION = 0.01;

class Tensor {

    public:

        // Constructor.
        Tensor(){}
        Tensor(const int nb_col, const int nb_row, const int wichInit = 0);

        // Destructor.
        ~Tensor(){}

        // Getter.
        std::vector<std::vector<double>> getTensor() const { return this->tensor; }

        // Setter.
        void setTensor(const std::vector<std::vector<double>> tensor) { this->tensor = tensor; }
        void addRow(const std::vector<double> row) { this->tensor.push_back(row); }
        
        // Addition.
        Tensor operator + (Tensor const &t2);
        void operator += (Tensor const &t2);

        // Substraction.
        Tensor operator - (Tensor const &t2);
        void operator -= (Tensor const &t2);

        // Division.
        Tensor operator / (Tensor const &t2);
        void operator /= (Tensor const &t2);
        void operator /= (double const &n);

        
        // Mulitiplication.
        Tensor operator * (Tensor const &t2);
        void operator *= (Tensor const &t2);
        void operator *= (double const &n);

        // Dot.
        double dot(Tensor const &t2);

        // Transposate.
        Tensor transposate();

        // Flatten.
        std::vector<double> flatten();

        // Operator.
        friend std::ostream& operator <<(std::ostream&, const Tensor&);

    private:
        std::vector<std::vector<double>> tensor;
};