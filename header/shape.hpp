#pragma once

#include <iostream>
#include <array>

class Shape {
    public:

        // Constructor.
        Shape(){}
        Shape(const int v1, const int v2) {this->shape = {v1, v2};}

        // Getter.
        std::array<int, 2> getShape() const { return this->shape; }

        // Cout.
        friend std::ostream& operator <<(std::ostream& out, const Shape& shape) {
            for (int &elt : shape.getShape()) {
                out << elt << " ";
            }
            out << "\n";
            return out;
        }

    private:
        std::array<int, 2> shape;
};