#ifndef GRAPHICS_H
#define GRAPHICS_H

#include <iostream>
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <SFML/Window.hpp>


#include "tensor_inline.hpp"

class Graphics {
    
    public:
        void show(const TensorInline& tensor, const int label);
};

#endif