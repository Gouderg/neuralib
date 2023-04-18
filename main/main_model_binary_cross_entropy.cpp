#include "main.hpp"

int main_model_binary_cross_entropy() {

    // Init the dataset.
    TensorInline X({NB_POINT * NB_LABEL_BINARY, NB_INPUTS}), y({1, NB_POINT * NB_LABEL_BINARY});
    TensorInline X_val({NB_POINT * NB_LABEL_BINARY, NB_INPUTS}), y_val({1, NB_POINT * NB_LABEL_BINARY});

    // Get the dataset.
    std::tie(X, y) = Dataset::spiral_data(NB_POINT, NB_LABEL_BINARY);
    std::tie(X_val, y_val) = Dataset::spiral_data(NB_POINT, NB_LABEL_BINARY);

    // Create the model.
    Model model;

    // Add all layers.
    model.add(new Layer_Dense({NB_INPUTS, NB_NEURON, WEIGHT_L1, WEIGHT_L2, BIAS_L1, BIAS_L2}));
    model.add(new Activation_ReLU());
    model.add(new Layer_Dense({NB_NEURON, 1}));
    model.add(new Activation_Sigmoid());           

    // Setup loss, optimizer and accuracy.
    Loss_BinaryCrossentropy loss_function;
    Optimizer_Adam optimizer = Optimizer_Adam(LEARNING_RATE, DECAY, MOMENTUM_EPSILON);
    Accuracy_Binary accuracy;


    model.set(&loss_function, &optimizer, &accuracy);

    model.train({.data={X, y}, 
                 .validatation_data={X_val, y_val}, 
                 .epochs=NB_EPOCH, 
                 .print_every=100,
                 .plotData=PlotConfiguration::circle});

    return 0;
}