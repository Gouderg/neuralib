#include "../header/tensor_inline.hpp"

// Constructor.
TensorInline::TensorInline(TensorInlineParams p) {

    assert(p.height > 0 && p.width > 0);

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

void reshape(const int new_height, const int new_width) {
    assert(new_height >= -1 && "Reshape height < -1");
    assert(new_width >= -1 && "Reshape width < -1");
    if (new_height == -1 && new_width == -1) { return; }

    // Si on tombe sur un -1, on prends la deuxième valeur et on la case n fois.
    // Il faut que la deuxième valeur soit divisible par le nombre total de valeur.


    /*
        old_w = 4
        old_h = 3
        w = -1
        h = 3

        if (old_w * old_h) % h  == 0

        new_h = 3
        new_w = old_w * old_h) // h
    */
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
    assert(this->height == t2.getHeight() && this->width == t2.getWidth() && "Wrong dimensions for +");


    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            t3.tensor[this->width * i + j] = this->tensor[this->width * i + j] + t2.tensor[this->width * i + j];
        }
    }

    return t3;
}

TensorInline TensorInline::operator + (double const &n) const {
    TensorInline t3 = *this;

    for (int i = 0; i < t3.getHeight() * t3.getWidth(); i++) {
        t3.tensor[i] += n;
    }

    return t3;
}

TensorInline operator + (const double &n, TensorInline const &t2) {
    TensorInline t3 = t2;

    for (int i = 0; i < t2.getHeight() * t2.getWidth(); i++) {
        t3.tensor[i] = n + t2.tensor[i];
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
    assert(this->height == t2.getHeight() && this->width == t2.getWidth() && "Wrong dimensions for +=");


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
    assert(this->height == t2.getHeight() && this->width == t2.getWidth() && "Wrong dimensions for -");


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

TensorInline operator - (const double &n, TensorInline const &t2) {
    TensorInline t3 = t2;

    for (int i = 0; i < t2.getHeight() * t2.getWidth(); i++) {
        t3.tensor[i] = n - t2.tensor[i];
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
    assert(this->height == t2.getHeight() && this->width == t2.getWidth() && "Wrong dimensions for -=");

    
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
    assert(this->height == t2.getHeight() && this->width == t2.getWidth() && "Wrong dimensions for *");


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

TensorInline operator * (const double &n, TensorInline const &t2) {
    TensorInline t3 = t2;

    for (int i = 0; i < t2.getHeight() * t2.getWidth(); i++) {
        t3.tensor[i] = n * t2.tensor[i];
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
    assert(this->height == t2.getHeight() && this->width == t2.getWidth() && "Wrong dimensions for *=");

    
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
    assert(this->height == t2.getHeight() && this->width == t2.getWidth() && "Wrong dimensions for /");


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
    assert(n != 0 && "Divide by 0");

    for (int i = 0; i < t3.getHeight() * getWidth(); i++) {
        t3.tensor[i] /= n;
    }

    return t3;
}

TensorInline operator / (const double &n, TensorInline const &t2) {
    TensorInline t3 = t2;

    for (int i = 0; i < t2.getHeight() * t2.getWidth(); i++) {
        t3.tensor[i] = t2.tensor[i] != 0 ? n / t2.tensor[i] : t2.tensor[i];
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
    assert(this->height == t2.getHeight() && this->width == t2.getWidth() && "Wrong dimensions for /=");
    
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
    assert(n != 0 && "Divide by 0");

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

bool TensorInline::operator <= (const double &n) const {
    for (int i = 0; i < this->width * this->height; i++) {
        if (this->tensor[i] > n)
            return false;
    }
    return true;
};

// Dot product.
TensorInline TensorInline::dot(const TensorInline& t1, const TensorInline& t2) {

    // Check conditions.
    assert(t1.getWidth() == t2.getHeight() && "Wrong dimensions for dot product."); 

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
    assert(t1.getWidth() == static_cast<int>(t2.size()) && "Wrong dimensions for dot product.");

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

// Exponential.
TensorInline TensorInline::exp(const TensorInline & t1) {
    TensorInline t = t1;
    for (int i = 0; i < static_cast<int>(t1.tensor.size()); i++) {
        t.tensor[i] = std::exp(t1.tensor[i]);
    }
    return t;
}

// Clipped the value in range.
TensorInline TensorInline::clipped(const TensorInline & t1, const double range_min, const double range_max) {
    TensorInline t_clip = t1;

    for (int i = 0; i < t_clip.getHeight() * t_clip.getWidth(); i++) {        
        if ((t_clip.tensor[i] < range_min)) {
            t_clip.tensor[i] = range_min;
        }

        if ((t_clip.tensor[i] > range_max)) {
            t_clip.tensor[i] = range_max;
        }
    }

    return t_clip;
}

// Binomial distribution.
TensorInline TensorInline::binomial(const TensorInlineBinomialParams p) {

    assert(p.trials >= 0 && "p.trials must be positive");
    assert(p.rate >= 0 && p.rate <= 1 && "p.rate must be in range [0; 1]");
    assert(p.height > 0 && "p.height must be positive");
    assert(p.width > 0 && "p.height must be positive");

    // Init binomial distribution.
    std::default_random_engine generator;
    std::binomial_distribution<int> distribution(p.trials, p.rate);

    TensorInline binome({p.height, p.width});
    for (int i = 0; i < p.height * p.width; i++) {
        binome.tensor[i] = distribution(generator);
    }

    return binome;
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