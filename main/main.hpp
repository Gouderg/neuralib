#ifndef MAIN_CUSTOM_H
#define MAIN_CUSTOM_H

#include "../header/activation_relu.hpp"
#include "../header/activation_sigmoid.hpp"
#include "../header/activation_softmax.hpp"
#include "../header/activation_linear.hpp"
#include "../header/dataset.hpp"
#include "../header/plot.hpp"
#include "../header/layer_dense.hpp"
#include "../header/layer_dropout.hpp"
#include "../header/loss.hpp"
#include "../header/activation_softmax_loss_categoricalcrossentropy.hpp"
#include "../header/optimizer.hpp"
#include "../header/statistic.hpp"
#include "../header/constants.hpp"

#include <ctime>

int main_binary_crossentropy();
int main_categorical_crossentropy();
int main_regression();
int main_debug();

#endif