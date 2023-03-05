#include "../header/tensor_inline.hpp"

// Constructor.
TensorInline::TensorInline(TensorInlineParams p) {

    if (p.height <= 0 || p.width <= 0) {
        std::cout << "Error TensorInline Constructor: cannot set tensor size <= 0" << std::endl;
        exit(1);
    }

    this->width = p.width;
    this->height = p.height;

    // Gaussian distribution.    
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(MEAN, STD_DEVIATION);
    srand(time(NULL)); 

    if (p.isRandom) {
        // Allocate space.
        this->tensor.reserve(p.height * p.width);
        
        // Gaussian distribution.    
        for (int i = 0; i < p.height * p.width; i++) {
            this->tensor.push_back(distribution(generator) * 0.02);
        }
    } else {
        this->tensor = std::vector<double> (p.height * p.width, p.valueToSet);
    }
}

// Addition.
TensorInline TensorInline::operator + (TensorInline const &t2) const{
    TensorInline t3({ this->height, t2.getWidth() });

    // Broadcasting operation.
    if (t2.getHeight() == 1 && t2.getWidth() == this->width) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                t3.tensor[this->width * i + j] = this->tensor[this->width * i + j] + t2.tensor[j];
            }
        }
        return t3;
    }

    // Other cases.
    if (this->height != t2.getHeight() || this->width != t2.getWidth()) { exit(1); return *this; }

    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            t3.tensor[this->width * i + j] = this->tensor[this->width * i + j] + t2.tensor[this->width * i + j];
        }
    }

    return t3;
}

TensorInline TensorInline::operator + (double const &n) const {
    TensorInline t3 = *this;

    for (int i = 0; i < t3.getHeight() * getWidth(); i++) {
        t3.tensor[i] += n;
    }

    return t3;
}

void TensorInline::operator += (TensorInline const &t2) {

    // Broadcasting operation.
    if (t2.getHeight() == 1 && t2.getWidth() == this->width) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                this->tensor[this->width * i + j] += t2.tensor[j];
            }
        }
        return ;
    }

    // Other cases.
    if (this->height != t2.getHeight() || this->width != t2.getWidth()) { exit(1); return ; }

    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            this->tensor[this->width * i + j] += t2.tensor[this->width * i + j];
        }
    }
}

void TensorInline::operator += (double const &n) {
    for (int i = 0; i < this->height * this->width; i++) {
        this->tensor[i] += n;
    }
}

// Substraction.
TensorInline TensorInline::operator - (TensorInline const &t2) const {
    TensorInline t3({ this->height, t2.getWidth() });

    // Broadcasting operation.
    if (t2.getHeight() == 1 && t2.getWidth() == this->width) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                t3.tensor[this->width * i + j] = this->tensor[this->width * i + j] - t2.tensor[j];
            }
        }
        return t3;
    }

    // Other cases.
    if (this->height != t2.getHeight() || this->width != t2.getWidth()) { return *this; }

    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            t3.tensor[this->width * i + j] = this->tensor[this->width * i + j] - t2.tensor[this->width * i + j];
        }
    }

    return t3;
}

TensorInline TensorInline::operator - (double const &n) const {
    TensorInline t3 = *this;

    for (int i = 0; i < t3.getHeight() * getWidth(); i++) {
        t3.tensor[i] -= n;
    }

    return t3;
}

void TensorInline::operator -= (TensorInline const &t2) {

    // Broadcasting operation.
    if (t2.getHeight() == 1 && t2.getWidth() == this->width) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                this->tensor[this->width * i + j] -= t2.tensor[j];
            }
        }
        return ;
    }

    // Other cases.
    if (this->height != t2.getHeight() || this->width != t2.getWidth()) { return ; }
    
    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            this->tensor[this->width * i + j] -= t2.tensor[this->width * i + j];
        }
    }
}

void TensorInline::operator -= (double const &n) {
    for (int i = 0; i < this->height * this->width; i++) {
        this->tensor[i] -= n;
    }
}

// Multiplication.
TensorInline TensorInline::operator * (TensorInline const &t2) const {
    TensorInline t3({ this->height, t2.getWidth() });

    // Broadcasting operation.
    if (t2.getHeight() == 1 && t2.getWidth() == this->width) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                t3.tensor[this->width * i + j] = this->tensor[this->width * i + j] * t2.tensor[j];
            }
        }
        return t3;
    }

    // Other cases.
    if (this->height != t2.getHeight() || this->width != t2.getWidth()) { exit(1); return *this; }

    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            t3.tensor[this->width * i + j] = this->tensor[this->width * i + j] * t2.tensor[this->width * i + j];
        }
    }

    return t3;
}

TensorInline TensorInline::operator * (double const &n) const {
    TensorInline t3 = *this;

    for (int i = 0; i < t3.getHeight() * getWidth(); i++) {
        t3.tensor[i] *= n;
    }

    return t3;
}

void TensorInline::operator *= (TensorInline const &t2) {

    // Broadcasting operation.
    if (t2.getHeight() == 1 && t2.getWidth() == this->width) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                this->tensor[this->width * i + j] *= t2.tensor[j];
            }
        }
        return ;
    }

    // Other cases.
    if (this->height != t2.getHeight() || this->width != t2.getWidth()) { exit(1); return ; }
    
    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            this->tensor[this->width * i + j] *= t2.tensor[this->width * i + j];
        }
    }
}

void TensorInline::operator *= (double const &n) {
    for (int i = 0; i < this->height * this->width; i++) {
        this->tensor[i] *= n;
    }
}

// Division.
TensorInline TensorInline::operator / (TensorInline const &t2) const {
    TensorInline t3({ this->height, t2.getWidth() });

    // Broadcasting operation.
    if (t2.getHeight() == 1 && t2.getWidth() == this->width) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                t3.tensor[this->width * i + j] = t2.tensor[j] == 0 ? this->tensor[this->width * i + j] : this->tensor[this->width * i + j] / t2.tensor[j] ;
            }
        }
        return t3;
    }

    // Other cases.
    if (this->height != t2.getHeight() || this->width != t2.getWidth()) { exit(1); return *this; }

    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            t3.tensor[this->width * i + j] = t2.tensor[this->width * i + j] == 0 ? this->tensor[this->width * i + j] : this->tensor[this->width * i + j] / t2.tensor[this->width * i + j] ;

        }
    }

    return t3;
}

TensorInline TensorInline::operator / (double const &n) const {
    TensorInline t3 = *this;

    // Division by 0.
    if (n == 0) { return t3; }

    for (int i = 0; i < t3.getHeight() * getWidth(); i++) {
        t3.tensor[i] /= n;
    }

    return t3;
}

void TensorInline::operator /= (TensorInline const &t2) {

    // Broadcasting operation.
    if (t2.getHeight() == 1 && t2.getWidth() == this->width) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                if (t2.tensor[j] != 0) {
                    this->tensor[this->width * i + j] /= t2.tensor[j];
                }
            }
        }
        return ;
    }

    // Other cases.
    if (this->height != t2.getHeight() || this->width != t2.getWidth()) { exit(1); return ; }
    
    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            if (t2.tensor[this->width * i + j] != 0) {
                this->tensor[this->width * i + j] /= t2.tensor[this->width * i + j];
            }
        }
    }
}

void TensorInline::operator /= (double const &n) {
    
    // Division by 0.
    if (n == 0) { return ;}

    for (int i = 0; i < this->height * this->width; i++) {
        this->tensor[i] /= n;
    }
}

// Comparaison.
bool TensorInline::operator == (TensorInline const &t2) const {
    return this->tensor == t2.tensor && this->width == t2.getWidth() && this->height == t2.getHeight();
}

bool TensorInline::operator != (TensorInline const &t2) const {
    return !(this->tensor == t2.tensor) || this->width != t2.getWidth() || this->height != t2.getHeight();
}

// Dot product.
TensorInline TensorInline::dot(const TensorInline& t1, const TensorInline& t2) {

    // Check conditions.
    if (t1.getWidth() != t2.getHeight()) {
        std::cout << "Wrong dimensions for dot product." << std::endl;
        return t1;
    }

    // Output vector.
    TensorInline t3 ({t1.getHeight(), t2.getWidth(), false, 0.0});

    const int w2 = t2.getWidth();
    const int w1 = t1.getWidth();
    int i, j, k;

    #pragma omp parallel for private(i,j,k) shared(t1, t2, t3) num_threads(nb_procs)
    for (i = 0; i < t1.getHeight(); i++) {
        for (k = 0; k < w1; k++) {
            for (j = 0; j < w2; j++) {
                t3.tensor[w2 * i + j] += t1.tensor[w1 * i + k] * t2.tensor[k * w2 + j];
            }
        }
    }
    return t3;
}

TensorInline TensorInline::dot(const TensorInline& t1, const std::vector<double>& t2) {

    // Check conditions.
    if (t1.getWidth() != static_cast<int>(t2.size())) {
        std::cout << "Wrong dimensions for dot product." << std::endl;
        return t1;
    }

    // Output vector.
    TensorInline t3 ({t1.getHeight(), static_cast<int>(t2.size()), false, 0.0});

    const int w2 = t2.size();
    const int w1 = t1.getWidth();
    int i, j, k;

    #pragma omp parallel for private(i,j,k) shared(t1, t2, t3) num_threads(nb_procs)
    for (i = 0; i < t1.getHeight(); i++) {
        for (k = 0; k < w1; k++) {
            for (j = 0; j < w2; j++) {
                t3.tensor[w2 * i + j] += t1.tensor[w1 * i + k] * t2[j];
            }
        }
    }
    return t3;
}

// Square root.
TensorInline TensorInline::sqrt() const {
    TensorInline t = *this;

    for (int i = 0; i < static_cast<int>(this->tensor.size()); i++) {
        t.tensor[i] =  this->tensor[i] >= 0.0 ? std::sqrt(this->tensor[i]) : this->tensor[i];
    }

    return t;
}

// Absolute value.
TensorInline TensorInline::abs() const {
    TensorInline t = *this;
    for (int i = 0; i < static_cast<int>(this->tensor.size()); i++) {
        t.tensor[i] = std::abs(this->tensor[i]);
    }

    return t;
}

// Transposate.
TensorInline TensorInline::transposate() const {
    TensorInline t3({this->width, this->height});

    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            t3.tensor[j * this->height + i] = this->tensor[this->width * i + j];
        }
    }
    return t3;
}

double TensorInline::sum(TensorInline const &t) {
    return std::accumulate(t.tensor.begin(), t.tensor.end(), 0.0);
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