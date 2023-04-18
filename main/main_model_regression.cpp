#include "main.hpp"


int main_model_regression() {
    // Init the dataset.
    TensorInline X({NB_REGRESSION_POINT, 1}), y({NB_REGRESSION_POINT, 1});
    TensorInline X_val({NB_REGRESSION_POINT, 1}), y_val({NB_REGRESSION_POINT, 1});

    // Get the dataset.
    std::tie(X, y) = Dataset::sine_data(NB_REGRESSION_POINT);
    std::tie(X_val, y_val) = Dataset::sine_data(NB_REGRESSION_POINT);

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

    model.train({.data={X, y}, 
                 .validatation_data={X_val, y_val}, 
                 .epochs=NB_EPOCH, 
                 .print_every=100});

    return 0;
}
