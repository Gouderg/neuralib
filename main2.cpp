#include "header/tensor.hpp"
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

const int NB_EPOCH = 1000;
const int NB_POINT = 500;
const int NB_NEURON = 64;

int main(int argc, char const *argv[]) {

    // Get the dataset.
    Tensor X, y;
    std::tie(X, y) = Dataset::spiral_data2(NB_POINT, 3);
    // std::tie(X, y) = Dataset::raw_value(100, 3);

    // Plot the dataset.
    // Plot plt;

    // plt.set_x_limit(-1, 1);
    // plt.set_y_limit(-1, 1);

    // for (int i = 0; i < X.shape().getY(); i++) {
    //     plt.draw_circle(X.getValue(i, 0), X.getValue(i, 1), 0.01 , Plot::getColor(y.getValue(0, i)));
    // }
    // plt.show();


    // Setup the statistic system.
    Statistic stat;

    // Create layer.
    Layer_Dense dense1(2, NB_NEURON, 0, 5e-4, 0, 5e-4);
    Layer_Dense dense2(NB_NEURON, 3);
    
    #ifdef MAIN1
    std::cout << "Utilisation de: \"Activation_Softmax_Loss_CategoricalCrossentropy\"" << std::endl;
    std::cout << "La fonction de perte et la dernière fonction d'activation sont combinées." << std::endl;


    // Activation function.
    Activation_ReLU activation1;

    // Loss function.
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;

    // Optimizer.
    // Optimizer_SGD optimizer = Optimizer_SGD(1.0, 1e-3, 0.9);
    // Optimizer_Adagrad optimizer = Optimizer_Adagrad(1.0, 1e-4, 1e-7);
    // Optimizer_RMSprop optimizer = Optimizer_RMSprop(0.02, 1e-5, 1e-7, 0.999);
    Optimizer_Adam optimizer = Optimizer_Adam(0.02, 5e-7);
    std::cout << "Algorithme de descente de gradient: " << optimizer << "\n\n" << std::endl;

    // Init stat value.
    double data_loss = 0.0, regularization_loss = 0.0, loss_val = 0.0, accuracy = 0.0;

    // Number of epoch.
    time_t start = std::time(NULL);
    for (int epoch = 0; epoch < NB_EPOCH; epoch++) {
        // Forward.
        dense1.forward(X);
        activation1.forward(dense1.getOutput());
        dense2.forward(activation1.getOutput());

        data_loss = loss_activation.forward(dense2.getOutput(), y);
        regularization_loss = loss_activation.getLoss().regularization_loss(dense1) + loss_activation.getLoss().regularization_loss(dense2);
        loss_val = data_loss + regularization_loss;
        accuracy = Loss::accuracy(loss_activation.getOutput(), y);


        // Backward.
        loss_activation.backward(loss_activation.getOutput(), y);
        dense2.backward(loss_activation.getDinputs());
        activation1.backward(dense2.getDinputs());
        dense1.backward(activation1.getDinputs());

        // Update weights and biases.
        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.post_update_params();

        // Get all the statistics.
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch;
            std::cout << ", acc: " << accuracy;
            std::cout << ", loss: " << loss_val;
            std::cout << ", (data_loss: " << data_loss;
            std::cout << ", regu_loss: " << regularization_loss;
            std::cout << "), lr: " << optimizer.getCurrentLr() << std::endl;
        }
        stat.update(loss_val, accuracy, optimizer.getCurrentLr());
    }
    time_t end = std::time(NULL);
    #else
    std::cout << "Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossEntropy" << std::endl;


    // Activation function.
    Activation_ReLU activation1;
    Activation_Softmax activation2;

    // Loss function.
    Loss_CategoricalCrossEntropy loss;

    // Forward.
    dense1.forward(X);
    activation1.forward(dense1.getOutput());
    dense2.forward(activation1.getOutput());
    activation2.forward(dense2.getOutput());

    double loss_val = loss.calculate(activation2.getOutput(), y);
    double accuracy = loss.accuracy(activation2.getOutput(), y);


    // Backward.
    loss.backward(activation2.getOutput(), y);
    activation2.backward(loss.getDinputs());
    dense2.backward(activation2.getDinputs());
    activation1.backward(dense2.getDinputs());
    dense1.backward(activation1.getDinputs());

    #endif
    std::cout << "Temps d'exécution: " << end - start << " sec." << std::endl;
    stat.plot(false);

    // Test our model.
    Tensor X_test, y_test;
    std::cout << "Test: " << std::endl;
    for (int i = 0; i < 10; i++) {
        std::tie(X_test, y_test) = Dataset::spiral_data2(NB_POINT, 3);

        // Forward.
        dense1.forward(X_test);
        activation1.forward(dense1.getOutput());
        dense2.forward(activation1.getOutput());

        double loss_val_test = loss_activation.forward(dense2.getOutput(), y_test);
        double accuracy_test = Loss::accuracy(loss_activation.getOutput(), y_test);
        std::cout << "Itérations n° " << i; 
        std::cout << ", loss: " << loss_val_test;
        std::cout << ", acc: " << accuracy_test << std::endl;
    }

    

    return 0;
}
