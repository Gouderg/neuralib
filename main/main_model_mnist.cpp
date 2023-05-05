#include "main.hpp"


int main_model_mnist() {

    // Load image
    TensorInline X = Dataset::read_idx_file("../datasets/fashionMNIST/train-images-idx3-ubyte", FileType::images);
    TensorInline y = Dataset::read_idx_file("../datasets/fashionMNIST/train-labels-idx1-ubyte", FileType::labels);
    TensorInline X_val = Dataset::read_idx_file("../datasets/fashionMNIST/t10k-images-idx3-ubyte", FileType::images);
    TensorInline y_val = Dataset::read_idx_file("../datasets/fashionMNIST/t10k-labels-idx1-ubyte", FileType::labels);

    // Scale image.
    Dataset::scale_pixels_values(X, ScaleFormat::betweenMinus1And1);
    Dataset::scale_pixels_values(X_val, ScaleFormat::betweenMinus1And1);

    // Create the model.
    Model model;

    // Add all layers.
    model.add(new Layer_Dense({X.getWidth(), NB_NEURON}));
    model.add(new Activation_ReLU());
    model.add(new Layer_Dense({NB_NEURON, NB_NEURON}));
    model.add(new Activation_ReLU());
    model.add(new Layer_Dense({NB_NEURON, 10}));
    model.add(new Activation_Softmax());

    // Setup loss, optimizer and accuracy.
    Loss_CategoricalCrossEntropy loss_function;
    Optimizer_Adam optimizer = Optimizer_Adam(LEARNING_RATE, 1e-3, MOMENTUM_EPSILON);
    Accuracy_Categorical accuracy(STRICT_ACCURACY_METRICS);


    model.set(&loss_function, &optimizer, &accuracy);

    model.finalize();

    model.train({.data={.X=X, .y=y}, 
                 .validatation_data={.X=X_val, .y=y_val}, 
                 .epochs=1, 
                 .print_every=100,
                 .batch_size=128,
                 .plotData=PlotConfiguration::none});

    return 0;
}