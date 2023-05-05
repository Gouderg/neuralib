#include "main.hpp"
#include "../header/dataset.hpp"
#include <ctime>
// g++ -fopenmp test.cpp ../src/dataset.cpp ../src/tools.cpp ../src/tensor_inline.cpp -o test && ./test

int test() {

    TensorInline X({11, 5, false, 2});
    TensorInline y({1, 11, false, 1});

    int batch_size = 2;

    int train_steps = X.getHeight() / batch_size;
    if (batch_size * train_steps < X.getHeight()) {
        train_steps += 1;
    }

    TensorInline batch_X({batch_size, X.getWidth()});
    TensorInline batch_y({1, batch_size});


    for (int step = 0; step < train_steps; step += 1) {
        if ((step+1) * batch_size > X.getHeight()) {
            batch_X.tensor.assign(X.tensor.begin() + step * batch_size * X.getWidth(), X.tensor.end());
            batch_y.tensor.assign(y.tensor.begin() + step * batch_size, y.tensor.end());
            std::cout << (step+1) * batch_size - batch_X.getHeight() << ", " << (step+1) * batch_size << ", " << batch_X.getHeight() << std::endl;
            std::cout << (step+1) * batch_size - batch_y.getWidth() << ", " << (step+1) * batch_size << ", " << batch_y.getWidth() << std::endl;

            batch_X.setHeight((step+1) * batch_size - X.getHeight());
            batch_y.setWidth((step+1) * batch_size - y.getWidth());
        } else {

            batch_X.tensor.assign(X.tensor.begin() + step * batch_size * X.getWidth(), X.tensor.begin() + (step+1) * batch_size * X.getWidth());
            batch_y.tensor.assign(y.tensor.begin() + step * batch_size, y.tensor.begin() + (step+1) * batch_size);
        }

        std::cout << batch_X;
        std::cout << batch_X.shape() << ", " << batch_X.tensor.size() << "\n";
        std::cout << batch_y;
        std::cout << batch_y.shape() << ", " << batch_y.tensor.size() << "\n" << std::endl;
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
