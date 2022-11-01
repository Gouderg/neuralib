#include "header/tensor.hpp"
#include "header/dataset.hpp"
#include "header/plot.hpp"

int main(int argc, char const *argv[]) {   

    Tensor X, y;
    Plot plt;

    plt.set_x_limit(-1, 1);
    plt.set_y_limit(-1, 1);


    std::tie(X, y) = Dataset::spiral_data(100, 3);

    for (int i = 0; i < X.getSizeY(); i++) {
        plt.draw_circle(X.getValue(i, 0), X.getValue(i, 1), 0.01 , Plot::getColor(y.getValue(0, i)));
    }
    plt.show();

    

    return 0;
}
