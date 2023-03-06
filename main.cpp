#include "header/dataset.hpp"
#include "header/plot.hpp"
#include "header/layer_dense.hpp"
#include "header/layer_dropout.hpp"
#include "header/activation_relu.hpp"
#include "header/activation_softmax.hpp"
#include "header/loss.hpp"
#include "header/activation_softmax_loss_categoricalcrossentropy.hpp"
#include "header/optimizer.hpp"
#include "header/statistic.hpp"
#include "header/constants.hpp"

#include <ctime>

#define TEST

int main() {

    // Get the dataset.
    TensorInline X({NB_POINT * NB_LABEL, NB_INPUTS}), y({1, NB_POINT * NB_LABEL});
    
    // Get the dataset.
    std::tie(X, y) = Dataset::spiral_data(NB_POINT, NB_LABEL);

    // Plot the dataset.
    // Plot plt;

    // plt.set_x_limit(-1, 1);
    // plt.set_y_limit(-1, 1);

    // for (int i = 0; i < X.getHeight() * X.getWidth(); i += 2) {
    //     plt.draw_circle(X.tensor[i], X.tensor[i + 1], Plot::getColor(y.tensor[static_cast<int>(i / 2)]));
    // }
    // plt.show();


    // Setup the statistic system.
    Statistic stat;

    // Create layer.
    Layer_Dense dense1(NB_INPUTS, NB_NEURON, WEIGHT_L1, WEIGHT_L2, BIAS_L1, BIAS_L2);
    Layer_Dense dense2(NB_NEURON, NB_LABEL);

    // Dropout layer.
    Layer_Dropout dropout1(DROPOUT_RATE);

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
    Optimizer_Adam optimizer = Optimizer_Adam(LEARNING_RATE, DECAY, MOMENTUM_EPSILON);
    std::cout << "Algorithme de descente de gradient: " << optimizer << "\n\n" << std::endl;

    // Init stat value.
    double data_loss = 0.0, regularization_loss = 0.0, loss_val = 0.0, accuracy = 0.0;

    // Number of epoch.
    time_t start = std::time(NULL);
    for (int epoch = 0; epoch < NB_EPOCH; epoch++) {

        // Forward.
        dense1.forward(X);
        activation1.forward(dense1.getOutput());
        dropout1.forward(activation1.getOutput());
        dense2.forward(dropout1.getOutput());

        data_loss = loss_activation.forward(dense2.getOutput(), y);
        regularization_loss = loss_activation.getLoss().regularization_loss(dense1) + loss_activation.getLoss().regularization_loss(dense2);
        loss_val = data_loss + regularization_loss;
        accuracy = Loss::accuracy(loss_activation.getOutput(), y);

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

        // Backward.
        loss_activation.backward(loss_activation.getOutput(), y);
        dense2.backward(loss_activation.getDinputs());
        dropout1.backward(dense2.getDinputs());
        activation1.backward(dropout1.getDinputs());
        dense1.backward(activation1.getDinputs());

        // Update weights and biases.
        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.post_update_params();

    }
    time_t end = std::time(NULL);

    std::cout << "Temps d'exécution: " << end - start << " sec." << std::endl;
    
    // Plot all the stats.
    stat.plot(false);

    // Test our model.
    TensorInline X_test({NB_POINT * NB_LABEL, NB_INPUTS}), y_test({1, NB_POINT * NB_LABEL});

    #ifdef TEST
    std::cout << "Test: " << std::endl;
    for (int i = 0; i < 10; i++) {
        std::tie(X_test, y_test) = Dataset::spiral_data(NB_POINT, NB_LABEL);

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
    #endif
    return 0;
}
