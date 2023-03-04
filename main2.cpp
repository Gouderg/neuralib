#include "header/dataset.hpp"
#include "header/plot.hpp"
#include "header/layer_dense.hpp"
#include "header/activation_relu.hpp"
#include "header/activation_softmax.hpp"
#include "header/loss.hpp"
#include "header/activation_softmax_loss_categoricalcrossentropy.hpp"
#include "header/optimizer.hpp"
#include "header/statistic.hpp"
#include <ctime>

#define MAIN1

const int NB_EPOCH = 1;
const int NB_POINT = 10;
const int NB_NEURON = 3;
const int NB_LABEL = 3;

int main() {

    // Get the dataset.
    TensorInline X({NB_POINT * NB_LABEL, 2}), y({1, NB_POINT * NB_LABEL});
    X.tensor = { 0.0,          0.,          0.10738789,  0.02852226,  0.09263825, -0.20199226, -0.32224888, -0.08524539, -0.3495118 ,  0.27454028, -0.52100587,  0.19285966,  0.5045865 ,  0.43570277,  0.76882404,  0.11767714,  0.49269393, -0.73984873, -0.70364994, -0.71054685, -0.        , -0.        , -0.07394107,  0.08293611,  0.00808054,  0.22207525,  0.24548167,  0.22549914,  0.38364738, -0.22437814, -0.00801609, -0.5554977 , -0.66060567,  0.08969161, -0.7174548 ,  0.30032802,  0.17299275,  0.87189275,  0.66193414,  0.74956197, -0        ,  0,  0.05838184, -0.09453698, -0.13682534, -0.17510438, -0.27516943, -0.18812999,  0.19194843,  0.40085742, -0.16649488,  0.53002024,  0.6666014 ,  0.00932745, 0.43282092, -0.6462231 , -0.87291753, -0.16774514, -0.6297623 ,  0.77678794};
    y.tensor = {0, 0, 0, 0, 0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2};

    Layer_Dense dense1(2, NB_NEURON, 0, 5e-4, 0, 5e-4);
    Layer_Dense dense2(NB_NEURON, NB_LABEL, 0, 5e-4, 0, 5e-4);

    dense1.setWeights({0.00154947, 0.00378163, -0.00887786, -0.01980796, -0.00347912,  0.00156349});
    dense2.setWeights({0.01230291,  0.0120238,  -0.00387327, -0.00302303, -0.01048553, -0.01420018, -0.0170627,   0.01950775, -0.00509652});

    Activation_ReLU activation1;
    Activation_Softmax activation2;

    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;



    dense1.forward(X);
    activation1.forward(dense1.getOutput());
    dense2.forward(activation1.getOutput());
    

    double loss_val = loss_activation.forward(dense2.getOutput(), y);
    double accuracy = Loss::accuracy(loss_activation.getOutput(), y);


    std::cout << loss_val << std::endl;
    std::cout << accuracy << std::endl;

    loss_activation.backward(loss_activation.getOutput(), y);
    dense2.backward(loss_activation.getDinputs());
    activation1.backward(dense2.getDinputs());
    dense1.backward(activation1.getDinputs());


    std::cout << dense1.getDweights() << std::endl;
    std::cout << dense1.getDbiases() << std::endl;
    std::cout << dense2.getDweights() << std::endl;
    std::cout << dense2.getDbiases() << std::endl;




   return 0;   
}