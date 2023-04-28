#include "main.hpp"
#include "../header/dataset.hpp"
#include <ctime>
// g++ -fopenmp test.cpp ../src/dataset.cpp ../src/tools.cpp ../src/tensor_inline.cpp -o test

int test() {

    TensorInline X({11, 5, false, 2});
    TensorInline y({11, 1, false, 1});

    int batch_size = 2;

    int train_steps = X.getHeight() / batch_size;
    if (batch_size * train_steps < X.getHeight()) {
        train_steps += 1;
    }

    TensorInline batch_X({batch_size, X.getWidth()});
    TensorInline batch_y({batch_size, y.getWidth()});

    std::cout << X << std::endl;
    std::cout << y << std::endl;
    for (int step = 0; step < train_steps; step += 1) {

        if ((step+1) * batch_size * X.getWidth() > X.getHeight() * X.getWidth()) {
            batch_X.tensor.assign(X.tensor.begin() + step * batch_size * X.getWidth(), X.tensor.end());
            batch_y.tensor.assign(y.tensor.begin() + step * batch_size * y.getWidth(), y.tensor.end());
            X.setHeight((step+1) * batch_size + X.getHeight());
            y.setHeight((step+1) * batch_size + y.getHeight());
        } else {

            batch_X.tensor.assign(X.tensor.begin() + step * batch_size * X.getWidth(), X.tensor.begin() + (step+1) * batch_size * X.getWidth());
            batch_y.tensor.assign(y.tensor.begin() + step * batch_size * y.getWidth(), y.tensor.begin() + (step+1) * batch_size * y.getWidth());
        }

        std::cout << batch_X << "\n" << batch_y << std::endl;

    }
    

    // time_t start = std::time(NULL);

    // // Load image
    // TensorInline y = Dataset::read_idx_file("datasets/fashionMNIST/train-labels-idx1-ubyte", FileType::labels);
    // TensorInline X = Dataset::read_idx_file("datasets/fashionMNIST/train-images-idx3-ubyte", FileType::images);

    // // Scale image.
    // Dataset::scale_pixels_values(X, ScaleFormat::betweenMinus1And1);

    // std::cout << y << std::endl;

    // time_t end = std::time(NULL);

    // std::cout << "Temps d'exécution: " << end - start << " sec." << std::endl;    
    
    return 0;
}
