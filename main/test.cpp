#include "header/dataset.hpp"
#include <ctime>
// g++ -fopenmp test.cpp src/dataset.cpp src/tools.cpp src/tensor_inline.cpp -o test

int main() {

    time_t start = std::time(NULL);

    // Load image
    TensorInline y = Dataset::read_idx_file("datasets/fashionMNIST/train-labels-idx1-ubyte", FileType::labels);
    TensorInline X = Dataset::read_idx_file("datasets/fashionMNIST/train-images-idx3-ubyte", FileType::images);

    // Scale image.
    Dataset::scale_pixels_values(X, ScaleFormat::betweenMinus1And1);

    std::cout << y << std::endl;

    time_t end = std::time(NULL);

    std::cout << "Temps d'exécution: " << end - start << " sec." << std::endl;    
    
    return 0;
}
