#pragma once


#include <iostream>
#include <vector>

class Matrix {

    public:

        // Product.
        static std::vector<std::vector<double>> product(std::vector<std::vector<double>> m1, std::vector<std::vector<double>> m2);      // Between two matrix.
        static std::vector<std::vector<double>> product(std::vector<std::vector<double>> m, double n);                                  // Between matrix and scalar.


        // Addition.
        static std::vector<std::vector<double>> add(std::vector<std::vector<double>> m1, std::vector<std::vector<double>> m2); 
        static std::vector<std::vector<double>> add(std::vector<std::vector<double>> m, double n);


        // Substraction.
        static std::vector<std::vector<double>> sub(std::vector<std::vector<double>> m1, std::vector<std::vector<double>> m2);
        static std::vector<std::vector<double>> sub(std::vector<std::vector<double>> m, double n);


        // Transposate matrix.
        static std::vector<std::vector<double>> transposate(std::vector<std::vector<double>> m);

};