#ifndef SHAPE
#define SHAPE

#include <iostream>
#include <array>

class Shape {
    public:

        // Constructor.
        Shape(){}
        Shape(const int v1, const int v2) {this->shape = {v1, v2};}

        // Getter.
        std::array<int, 2> getShape() const { return this->shape; }
        int getX() const { return this->shape[1]; }
        int getY() const { return this->shape[0]; }


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

#endif