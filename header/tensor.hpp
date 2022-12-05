#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <functional>
#include <ctime>

#include "shape.hpp"

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
        double getValue(const int i, const int j) const { return this->tensor[i][j]; }
        std::vector<double> getRow(const int i) const { return this->tensor[i]; }
        
        // Setter.
        void setTensor(const std::vector<std::vector<double>> tensor) { this->tensor = tensor; }
        void addRow(const std::vector<double> row) { this->tensor.push_back(row); }
        void setValue(const int i, const int j, const double value) { this->tensor[i][j] = value; }
        void setRow(const int i, const std::vector<double> value) { this->tensor[i] = value; }


        
        // Addition.
        Tensor operator + (Tensor const &t2);
        Tensor operator + (double const &n);
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
        Tensor operator * (double const &n);
        void operator *= (Tensor const &t2);
        void operator *= (double const &n);

        // Dot.
        Tensor dot(Tensor const &t2);
        static Tensor dot(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2);
        std::vector<double> dot(std::vector<double> v1);


        // Transposate.
        Tensor transposate();
        static std::vector<std::vector<double>> transposate(std::vector<std::vector<double>> v1);

        // Flatten.
        std::vector<double> flatten();

        // Cout.
        friend std::ostream& operator <<(std::ostream&, const Tensor&);

        // Shape.
        Shape shape() { return Shape(static_cast<int>(this->tensor.size()), static_cast<int>(this->tensor[0].size())); }
        int shapeX() { return shape().getX(); }
        int shapeY() { return shape().getY(); }


    private:
        std::vector<std::vector<double>> tensor;
};

#endif