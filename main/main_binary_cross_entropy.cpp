#include "main.hpp"

#define TEST
// #define PLOT

int main_binary_crossentropy() {

    // Get the dataset.
    TensorInline X({NB_POINT * NB_LABEL_BINARY, NB_INPUTS}), y({1, NB_POINT * NB_LABEL_BINARY});
    
    // Get the dataset.
    std::tie(X, y) = Dataset::spiral_data(NB_POINT, NB_LABEL_BINARY);
    y.reshape(-1, 1);

    #ifdef PLOT
    // Plot the dataset.
    Plot plt;

    plt.set_x_limit(-1, 1);
    plt.set_y_limit(-1, 1);

    for (int i = 0; i < X.getHeight() * X.getWidth(); i += 2) {
        plt.draw_circle(X.tensor[i], X.tensor[i + 1], Plot::getColor(y.tensor[static_cast<int>(i / 2)]));
    }
    plt.show();
    #endif

    // Setup the statistic system.
    Statistic stat;

    // Create layer.
    Layer_Dense dense1(NB_INPUTS, NB_NEURON, WEIGHT_L1, WEIGHT_L2, BIAS_L1, BIAS_L2);
    Layer_Dense dense2(NB_NEURON, NB_LABEL_BINARY);


    std::cout << "Utilisation de: \"Binary Logistic Regression\"" << std::endl;

    // Activation function.
    Activation_ReLU activation1;
    Activation_Sigmoid activation2;

    // Loss function.
    Loss_BinaryCrossentropy loss_function;

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
        dense2.forward(activation1.getOutput());
        activation2.forward(dense2.getOutput());

        data_loss = loss_function.calculate(activation2.getOutput(), y);
        regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2);
        loss_val = data_loss + regularization_loss;

        accuracy = Loss_BinaryCrossentropy::accuracy(activation2.getOutput(), y);
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
        loss_function.backward(activation2.getOutput(), y);
        activation2.backward(loss_function.getDinputs());
        dense2.backward(activation2.getDinputs());
        activation1.backward(dense2.getDinputs());
        dense1.backward(activation1.getDinputs());

        // Update weights and biases.
        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.post_update_params();

    }
    time_t end = std::time(NULL);

    std::cout << "Temps d'exécution: " << end - start << " sec." << std::endl;
    
    #ifdef TEST
    // Plot all the stats.
    stat.plot(false);

    // Test our model.
    TensorInline X_test({NB_POINT * NB_LABEL_BINARY, NB_INPUTS}), y_test({1, NB_POINT * NB_LABEL_BINARY});

    std::cout << "Test: " << std::endl;
    for (int i = 0; i < 10; i++) {
        std::tie(X_test, y_test) = Dataset::spiral_data(NB_POINT, NB_LABEL_BINARY);
        y_test.reshape(-1, 1);


        // Forward.
        dense1.forward(X_test);
        activation1.forward(dense1.getOutput());
        dense2.forward(activation1.getOutput());
        activation2.forward(dense2.getOutput());

        double loss_val_test = loss_function.calculate(activation2.getOutput(), y_test);
        double accuracy_test = Loss_BinaryCrossentropy::accuracy(activation2.getOutput(), y_test);
        std::cout << "Itérations n° " << i; 
        std::cout << ", loss: " << loss_val_test;
        std::cout << ", acc: " << accuracy_test << std::endl;
    }
    #endif

    return 0;
}
