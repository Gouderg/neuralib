#include "main.hpp"

#define PLOT
#define TEST

int main_regression() {

    // Get the dataset.
    TensorInline X({1000, 1}), y({1000, 1});
    
    // Get the dataset.
    std::tie(X, y) = Dataset::sine_data(1000);

    #ifdef PLOT
    // Plot the dataset.
    Plot plt;

    plt.set_x_limit(-1, 1);
    plt.set_y_limit(-1, 1);

    plt.draw_line(y.tensor, "red");

    #endif

    // Setup the statistic system.
    Statistic stat;

    // Create layer.
    Layer_Dense dense1(1, NB_NEURON);
    Layer_Dense dense2(NB_NEURON, NB_NEURON);
    Layer_Dense dense3(NB_NEURON, 1);



    std::cout << "Utilisation de: \"Binary Logistic Regression\"" << std::endl;

    // Activation function.
    Activation_ReLU activation1;
    Activation_ReLU activation2;
    Activation_Linear activation3;

    // Loss function.
    Loss_MeanSquaredError loss_function;

    // Optimizer.
    // Optimizer_SGD optimizer = Optimizer_SGD(1.0, 1e-3, 0.9);
    // Optimizer_Adagrad optimizer = Optimizer_Adagrad(1.0, 1e-4, 1e-7);
    // Optimizer_RMSprop optimizer = Optimizer_RMSprop(0.02, 1e-5, 1e-7, 0.999);
    Optimizer_Adam optimizer = Optimizer_Adam(0.005, 1e-3, MOMENTUM_EPSILON);
    std::cout << "Algorithme de descente de gradient: " << optimizer << "\n\n" << std::endl;

    // Init stat value.
    double data_loss = 0.0, regularization_loss = 0.0, loss_val = 0.0, accuracy = 0.0;
    double accuracy_precision = TensorInline::standard_deviation(y) / STRICT_ACCURACY_METRICS;

    // Number of epoch.
    time_t start = std::time(NULL);
    for (int epoch = 0; epoch < NB_EPOCH; epoch++) {

        // Forward.
        dense1.forward(X);
        activation1.forward(dense1.getOutput());
        dense2.forward(activation1.getOutput());
        activation2.forward(dense2.getOutput());
        dense3.forward(activation2.getOutput());
        activation3.forward(dense3.getOutput());

        data_loss = loss_function.calculate(activation3.getOutput(), y);
        regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2) + loss_function.regularization_loss(dense3);
        loss_val = data_loss + regularization_loss;

        accuracy = Loss_MeanSquaredError::accuracy(activation3.getOutput(), y, accuracy_precision);
        
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
        loss_function.backward(activation3.getOutput(), y);
        activation3.backward(loss_function.getDinputs());
        dense3.backward(activation3.getDinputs());
        activation2.backward(dense3.getDinputs());
        dense2.backward(activation2.getDinputs());
        activation1.backward(dense2.getDinputs());
        dense1.backward(activation1.getDinputs());


        // Update weights and biases.
        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.update_params(dense3);
        optimizer.post_update_params();

    }
    time_t end = std::time(NULL);

    std::cout << "Temps d'exécution: " << end - start << " sec." << std::endl;

    #ifdef TEST
    // Plot all the stats.
    stat.plot(false);

    // Test our model.
    TensorInline X_test({1000, 1}), y_test({1000, 1});

    std::cout << "Test: " << std::endl;

    std::tie(X_test, y_test) = Dataset::sine_data(1000);

    // Forward.
    dense1.forward(X_test);
    activation1.forward(dense1.getOutput());
    dense2.forward(activation1.getOutput());
    activation2.forward(dense2.getOutput());
    dense3.forward(activation2.getOutput());
    activation3.forward(dense3.getOutput());

    Plot plt_test;

    plt_test.draw_line(activation3.getOutput().tensor, "red");
    
    #endif

    return 0;
}
