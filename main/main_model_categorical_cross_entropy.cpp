#include "main.hpp"


int main_model_categorical_cross_entropy() {

    // Get the dataset.
    Data d = Dataset::spiral_data(NB_POINT, NB_LABEL_CATEGORICAL);
    Data d_val = Dataset::spiral_data(NB_POINT, NB_LABEL_CATEGORICAL);

    // Create the model.
    Model model;

    // Add all layers.
    model.add(new Layer_Dense({NB_INPUTS, NB_NEURON, WEIGHT_L1, WEIGHT_L2, BIAS_L1, BIAS_L2}));
    model.add(new Activation_ReLU());
    model.add(new Layer_Dropout(DROPOUT_RATE));
    model.add(new Layer_Dense({NB_NEURON, NB_LABEL_CATEGORICAL}));
    model.add(new Activation_Softmax());

    // Setup loss, optimizer and accuracy.
    Loss_CategoricalCrossEntropy loss_function;
    Optimizer_Adam optimizer = Optimizer_Adam(LEARNING_RATE, DECAY, MOMENTUM_EPSILON);
    Accuracy_Categorical accuracy(STRICT_ACCURACY_METRICS);


    model.set(&loss_function, &optimizer, &accuracy);

    model.finalize();

    model.train({.data=d, 
                 .validatation_data=d_val, 
                 .epochs=NB_EPOCH, 
                 .print_every=100,
                 .plotData=PlotConfiguration::circle});

    return 0;
}