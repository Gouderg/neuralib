#include "header/tensor.hpp"
#include "header/dataset.hpp"
#include "header/plot.hpp"
#include "header/layer_dense.hpp"

int main(int argc, char const *argv[]) {   
    
    // Get the dataset.
    Tensor X, y;
    std::tie(X, y) = Dataset::spiral_data(100, 3);

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
    dense1.forward(X);

    std::cout << dense1.getOutput() << std::endl;


    

    return 0;
}
