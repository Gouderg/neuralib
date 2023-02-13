#include "../header/tensor_inline.hpp"

// Constructor.
TensorInline::TensorInline(const int nb_row, const int nb_col, const int whichInit) {

    this->width = nb_col;
    this->height = nb_row;

    // Gaussian distribution.    
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(MEAN, DEVIATION);
    srand(time(NULL)); 

    switch(whichInit) {
        case 2:
            this->tensor = std::vector<double> (nb_row * nb_col, 1.0);
            break;

        case 1:
            // Allocate space.
            this->tensor.reserve(nb_row * nb_col);
            
            // Gaussian distribution.    
            for (int i = 0; i < nb_row * nb_col; i++) {
                this->tensor.push_back(distribution(generator));
            }
            break;

        case 0: default:
            this->tensor = std::vector<double> (nb_row * nb_col, 0.0);
            break;
    }   
}

// Cout.
std::ostream& operator<<(std::ostream& out, const TensorInline& t) {

    const int w = t.getWidth();
    for (int i = 0; i < t.getHeight(); i++) {
        for (int j = 0; j < w; j++) {
            out << t.tensor[i * w + j] << " ";
        }
        out << "\n";
    }
    return out;
}

// Dot product.
TensorInline TensorInline::dot(const TensorInline& t2) {

    // Check conditions.
    if (this->width != t2.getHeight()) {
        std::cout << "Wrong dimensions for dot product." << std::endl;
    }

    // Output vector.
    TensorInline t3 = TensorInline(this->height, t2.getWidth(), 0);

    const int w = t2.getWidth();
    int i, j, k;
    TensorInline t1 = *this;
    #pragma omp parallel for private(i,j,k) shared(t1, t2, t3)
    for (i = 0; i < this->height; i++) {
        for (k = 0; k < this->width; k++) {
            for (j = 0; j < w; j++) {
                t3.tensor[w * i + j] += t1.tensor[this->width * i + k] * t2.tensor[k * w + j];
            }
        }
    }
    return t3;
}

// Dot product.
TensorInline TensorInline::dot(const TensorInline& t1, const TensorInline& t2) {

    // Check conditions.
    if (t1.getWidth() != t2.getHeight()) {
        std::cout << "Wrong dimensions for dot product." << std::endl;
    }

    // Output vector.
    TensorInline t3 = TensorInline(t1.getHeight(), t2.getWidth(), 0);

    const int w2 = t2.getWidth();
    const int w1 = t1.getWidth();
    int i, j, k;

    #pragma omp parallel for private(i,j,k) shared(t1, t2, t3)
    for (i = 0; i < t1.getHeight(); i++) {
        for (k = 0; k < w1; k++) {
            for (j = 0; j < w2; j++) {
                t3.tensor[w2 * i + j] += t1.tensor[w1 * i + k] * t2.tensor[k * w2 + j];
            }
        }
    }
    return t3;
}