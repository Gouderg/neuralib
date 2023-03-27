#include "../header/layer.hpp"

void Layer_Input::forward(const TensorInline& inputs, const bool training) {
    this->output = inputs;
}