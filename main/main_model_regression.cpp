#include "main.hpp"


int main_model_regression() {

    // Get the dataset.
    Data d = Dataset::sine_data(NB_REGRESSION_POINT);
    Data d_val = Dataset::sine_data(NB_REGRESSION_POINT);

    // Create the model.
    Model model;

    // Add all layers.
    model.add(new Layer_Dense({.n_inputs=1, .n_neurons=NB_NEURON, .randomFactor=RANDOM_REGRESSION_FACTOR}));
    model.add(new Activation_ReLU());
    model.add(new Layer_Dense({.n_inputs=NB_NEURON, .n_neurons=NB_NEURON, .randomFactor=RANDOM_REGRESSION_FACTOR}));
    model.add(new Activation_ReLU());
    model.add(new Layer_Dense({.n_inputs=NB_NEURON, .n_neurons=1, .randomFactor=RANDOM_REGRESSION_FACTOR}));
    model.add(new Activation_Linear());           

    // Setup loss, optimizer and accuracy.
    Loss_MeanSquaredError loss_function;
    Optimizer_Adam optimizer = Optimizer_Adam(0.005, 1e-3, MOMENTUM_EPSILON);
    Accuracy_Regression accuracy(STRICT_ACCURACY_METRICS);


    model.set(&loss_function, &optimizer, &accuracy);

    model.finalize();

    model.train({.data=d, 
                 .validatation_data=d_val, 
                 .epochs=NB_EPOCH, 
                 .print_every=100});

    return 0;
}
