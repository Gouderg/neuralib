#include "main.hpp"

#define TEST
#define PLOT

int main_categorical_crossentropy() {

    
    // Get the dataset.
    Data d = Dataset::spiral_data(NB_POINT, NB_LABEL_CATEGORICAL);

    #ifdef PLOT
    // Plot the dataset.
    Plot plt;

    plt.set_x_limit(-1, 1);
    plt.set_y_limit(-1, 1);

    for (int i = 0; i < d.X.getHeight() * d.X.getWidth(); i += 2) {
        plt.draw_circle(d.X.tensor[i], d.X.tensor[i + 1], Plot::getColor(d.y.tensor[static_cast<int>(i / 2)]));
    }
    plt.show();
    #endif

    // Setup the statistic system.
    Statistic stat;

    // Create layer.
    Layer_Dense dense1({NB_INPUTS, NB_NEURON, WEIGHT_L1, WEIGHT_L2, BIAS_L1, BIAS_L2});
    Layer_Dense dense2({NB_NEURON, NB_LABEL_CATEGORICAL});

    // Dropout layer.
    Layer_Dropout dropout1(DROPOUT_RATE);

    std::cout << "Utilisation de: \"Activation_Softmax_Loss_CategoricalCrossentropy\"" << std::endl;
    std::cout << "La fonction de perte et la dernière fonction d'activation sont combinées." << std::endl;


    // Activation function.
    Activation_ReLU activation1;

    // Loss function.
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;

    // Accuracy.
    Accuracy_Categorical accuracy_function(false);
    accuracy_function.init(d.y);

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
        dense1.forward(d.X, true);
        activation1.forward(dense1.getOutput(), true);
        dropout1.forward(activation1.getOutput(), true);
        dense2.forward(dropout1.getOutput(), true);
        
        data_loss = loss_activation.forward(dense2.getOutput(), d.y);
        regularization_loss = loss_activation.getLoss().regularization_loss(dense1) + loss_activation.getLoss().regularization_loss(dense2);
        loss_val = data_loss + regularization_loss;
        accuracy = accuracy_function.calculate(loss_activation.getOutput(), d.y);

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
        loss_activation.backward(loss_activation.getOutput(), d.y);
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
    
    #ifdef TEST
    // Plot all the stats.
    stat.plot(false);

    // Test our model.

    std::cout << "Test: " << std::endl;
    for (int i = 0; i < 10; i++) {
        Data d_test = Dataset::spiral_data(NB_POINT, NB_LABEL_CATEGORICAL);

        // Forward.
        dense1.forward(d_test.X);
        activation1.forward(dense1.getOutput());
        dense2.forward(activation1.getOutput());

        double loss_val_test = loss_activation.forward(dense2.getOutput(), d_test.y);
        double accuracy_test = accuracy_function.calculate(loss_activation.getOutput(), d_test.y);
        std::cout << "Itérations n° " << i; 
        std::cout << ", loss: " << loss_val_test;
        std::cout << ", acc: " << accuracy_test << std::endl;
    }
    #endif

    return 0;
}
