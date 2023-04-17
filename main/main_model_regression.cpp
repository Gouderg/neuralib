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
    model.add(new Layer_Dense({1, NB_NEURON}));
    model.add(new Activation_ReLU());
    model.add(new Layer_Dense({NB_NEURON, NB_NEURON}));
    model.add(new Activation_ReLU());
    model.add(new Layer_Dense({NB_NEURON, 1}));
    model.add(new Activation_Linear());           

    // Setup loss, optimizer and accuracy.
    Loss_MeanSquaredError loss_function;
    Optimizer_Adam optimizer = Optimizer_Adam(0.005, 1e-3, MOMENTUM_EPSILON);
    Accuracy_Regression accuracy(STRICT_ACCURACY_METRICS);


    model.set(&loss_function, &optimizer, &accuracy);

    model.train({std::make_tuple(X, y), std::make_tuple(X_val, y_val), NB_EPOCH, 100});

    return 0;
}
