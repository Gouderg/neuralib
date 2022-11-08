#include "header/tensor.hpp"
#include "header/dataset.hpp"
#include "header/plot.hpp"
#include "header/layer_dense.hpp"
#include "header/activation_relu.hpp"
#include "header/activation_softmax.hpp"
#include "header/loss.hpp"
#include "header/activation_softmax_loss_categoricalcrossentropy.hpp"

int main(int argc, char const *argv[]) {

    // Get the dataset.
    Tensor X, y;
    // std::tie(X, y) = Dataset::spiral_data(100, 3);
    std::tie(X, y) = Dataset::raw_value(100, 3);

    // Plot the dataset
    // Plot plt;

    // plt.set_x_limit(-1, 1);
    // plt.set_y_limit(-1, 1);

    // for (int i = 0; i < X.shape().getY(); i++) {
    //     plt.draw_circle(X.getValue(i, 0), X.getValue(i, 1), 0.01 , Plot::getColor(y.getValue(0, i)));
    // }
    // plt.show();

    // Create layer.
    Layer_Dense dense1(2, 3);
    std::cout << dense1.getWeights() << std::endl;
    dense1.setWeight({{-0.01306527, 0.01658131, -0.00118164}, {-0.00680178, 0.00666383, -0.0046072 }});
    Layer_Dense dense2(3, 3);
    dense2.setWeight({{-0.01334258, -0.01346717, 0.00693773}, {-0.00159573, -0.00133702,  0.01077744}, {-0.01126826, -0.00730678, -0.0038488 }});

    // Activation function.
    Activation_ReLU activation1;
    Activation_Softmax activation2;

    // Loss function.
    Loss_CategoricalCrossEntropy loss;
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;

    // Forward.
    dense1.forward(X);
    activation1.forward(dense1.getOutput());
    dense2.forward(activation1.getOutput());
    // activation2.forward(dense2.getOutput());

    double loss_val = loss_activation.forward(dense2.getOutput(), y);
    double accuracy = loss.accuracy(loss_activation.getOutput(), y);

    // double loss_val = loss.calculate(activation2.getOutput(), y);
    // double accuracy = loss.accuracy(activation2.getOutput(), y);


    std::cout << "loss: " << loss_val << std::endl;
    std::cout << "acc: " << accuracy << std::endl;


    // Backward.

    // loss.backward(activation2.getOutput(), y);
    // activation2.backward(loss.getDinputs());
    loss_activation.backward(loss_activation.getOutput(), y);
    dense2.backward(loss_activation.getDinputs());
    // std::cout << dense2.getDinputs() << std::endl;
    activation1.backward(dense2.getDinputs());
    dense1.backward(activation1.getDinputs());
    std::cout.precision(17);
    std::cout << std::fixed << dense1.getDweights() << std::endl;
    std::cout << dense1.getDbiases() << std::endl;

    return 0;
}
