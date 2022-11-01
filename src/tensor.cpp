#include "../header/tensor.hpp"


// Constructor.
Tensor::Tensor(const int nb_col, const int nb_row, const int wichInit) {

    // Gaussian distribution.    
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(MEAN, DEVIATION);
    srand(time(NULL)); 
    switch (wichInit) {
        case 1:
            
            for (int i = 0; i < nb_col; i++) {
                std::vector<double> v1;
                for (int j = 0; j < nb_row; j++) {
                    v1.push_back(distribution(generator));
                }
                this->tensor.push_back(v1);
            }
            break;
        
        case 0: default:
            for (int i = 0; i < nb_col; i++) {
                std::vector<double> v1(nb_row, 2);
                this->tensor.push_back(v1);
            }
            break;
    }
}

// Add.
Tensor Tensor::operator + (Tensor const &t2) {
    std::vector<std::vector<double>> t2_tensor = t2.getTensor();
    Tensor t3(this->tensor.size(), t2_tensor[0].size());


    // Broadcasting addition.
    if (t2_tensor.size() == 1 && this->tensor[0].size() == t2_tensor[0].size()) {
        for (int i = 0; i < this->tensor.size(); i++) {
            for (int j = 0; j < t2_tensor[0].size(); j++) {
                t3.addValue(i, j, this->tensor[i][j] + t2_tensor[0][j]);
            }
        }
        return t3;
    }
    

    // Other case.
    if (this->tensor.size() != t2_tensor.size() || this->tensor[0].size() != t2_tensor[0].size()) { return *this; }

    for (int i = 0; i < this->tensor.size(); i++) {
        for (int j = 0; j < this->tensor[i].size(); j++) {
            t3.addValue(i, j, this->tensor[i][j] + t2_tensor[i][j]);
        }
    }

    return t3;
}

void Tensor::operator += (Tensor const &t2) {

    std::vector<std::vector<double>> t2_tensor = t2.getTensor();

    if (this->tensor.size() != t2_tensor.size() || this->tensor[0].size() != t2_tensor[0].size()) { return ; }

    for (int i = 0; i < this->tensor.size(); i++) {
        for (int j = 0; j < this->tensor[i].size(); j++) {
            this->tensor[i][j] += t2_tensor[i][j];
        }
    }
}

Tensor Tensor::operator + (double const &n) {
    Tensor t;
    for (int i = 0; i < this->tensor.size(); i++) {
        std::vector<double> v1;
        for (int j = 0; j < this->tensor[i].size(); j++) {
            v1.push_back(this->tensor[i][j] + n);
        }
        t.addRow(v1);
    }
    return t;
}

// Substraction.
Tensor Tensor::operator - (Tensor const &t2) {
    Tensor t3;
    std::vector<std::vector<double>> t2_tensor = t2.getTensor();

    if (this->tensor.size() != t2_tensor.size() || this->tensor[0].size() != t2_tensor[0].size()) { return *this; }

    for (int i = 0; i < this->tensor.size(); i++) {
        std::vector<double> v1;
        for (int j = 0; j < this->tensor[i].size(); j++) {
            v1.push_back(this->tensor[i][j] - t2_tensor[i][j]);
        }
        t3.addRow(v1);
    }

    return t3;
}

void Tensor::operator -= (Tensor const &t2) {

    std::vector<std::vector<double>> t2_tensor = t2.getTensor();

    if (this->tensor.size() != t2_tensor.size() || this->tensor[0].size() != t2_tensor[0].size()) { return ; }

    for (int i = 0; i < this->tensor.size(); i++) {
        for (int j = 0; j < this->tensor[i].size(); j++) {
            this->tensor[i][j] -= t2_tensor[i][j];
        }
    }
}

// Division.
Tensor Tensor::operator / (Tensor const &t2) {
    Tensor t3;
    std::vector<std::vector<double>> t2_tensor = t2.getTensor();

    if (this->tensor.size() != t2_tensor.size() || this->tensor[0].size() != t2_tensor[0].size()) { return *this; }

    for (int i = 0; i < this->tensor.size(); i++) {
        std::vector<double> v1;
        for (int j = 0; j < this->tensor[i].size(); j++) {
            v1.push_back(t2_tensor[i][j] != 0 ? this->tensor[i][j] / t2_tensor[i][j] : this->tensor[i][j]);
        }
        t3.addRow(v1);
    }

    return t3;
}

void Tensor::operator /= (Tensor const &t2) {

    std::vector<std::vector<double>> t2_tensor = t2.getTensor();

    if (this->tensor.size() != t2_tensor.size() || this->tensor[0].size() != t2_tensor[0].size()) { return ; }

    for (int i = 0; i < this->tensor.size(); i++) {
        for (int j = 0; j < this->tensor[i].size(); j++) {
            if (t2_tensor[i][j] != 0) {
                this->tensor[i][j] /= t2_tensor[i][j];
            }
        }
    }
}

void Tensor::operator /= (double const &n) {
    if (n == 0) { return ; }
    for (int i = 0; i < this->tensor.size(); i++) {
        for (int j = 0; j < this->tensor[i].size(); j++) {
            this->tensor[i][j] /= n;
        }
    }
}

// Multiplication.
Tensor Tensor::operator * (Tensor const &t2) {
    Tensor t3;
    std::vector<std::vector<double>> t2_tensor = t2.getTensor();

    if (this->tensor.size() != t2_tensor.size() || this->tensor[0].size() != t2_tensor[0].size()) { return *this; }

    for (int i = 0; i < this->tensor.size(); i++) {
        std::vector<double> v1;
        for (int j = 0; j < this->tensor[i].size(); j++) {
            v1.push_back(this->tensor[i][j] * t2_tensor[i][j]);
        }
        t3.addRow(v1);
    }

    return t3;
}

void Tensor::operator *= (Tensor const &t2) {

    std::vector<std::vector<double>> t2_tensor = t2.getTensor();

    if (this->tensor.size() != t2_tensor.size() || this->tensor[0].size() != t2_tensor[0].size()) { return ; }

    for (int i = 0; i < this->tensor.size(); i++) {
        for (int j = 0; j < this->tensor[i].size(); j++) {
            this->tensor[i][j] *= t2_tensor[i][j];
        }
    }
}

void Tensor::operator *= (double const &n) {
    for (int i = 0; i < this->tensor.size(); i++) {
        for (int j = 0; j < this->tensor[i].size(); j++) {
            this->tensor[i][j] *= n;
        }
    }
}

// Dot.
Tensor Tensor::dot(Tensor const &t2) {

    std::vector<std::vector<double>> t2_tensor = t2.getTensor();

    // If the dimensions mismatch.
    if (this->tensor.size() != t2_tensor[0].size() || this->tensor[0].size() != t2_tensor.size()) { return *this; }

    Tensor output(this->tensor.size(), t2_tensor[0].size());
    double val = 0;

    for (int i = 0; i < this->tensor.size(); i++) {
        for (int j = 0; j < t2_tensor[0].size();j++) {
            val = 0;
            for (int k = 0; k < this->tensor[0].size(); k++) {
                val += this->tensor[i][k] * t2_tensor[k][j];
            }
            output.addValue(i, j, val);
        }
    }
    
    return output;
}

// Transposate.
Tensor Tensor::transposate() {
    Tensor output;

    // Run through each column.
    for (int j = 0; j < this->tensor[0].size(); j++) {
        std::vector<double> line_output;
        
        // Run through each line.
        for (int i = 0; i < this->tensor.size(); i++) {
           line_output.push_back(this->tensor[i][j]);
        }

        output.addRow(line_output);
    }
    
    return output;
}

// Flatten.
std::vector<double> Tensor::flatten() {
    std::vector<double> output;

    for (auto elt : this->tensor) {
        output.insert(output.end(), elt.begin(), elt.end());
    }

    return output;
}

// Cout.
std::ostream& operator<<(std::ostream& out, const Tensor& tensor) {
    for (auto &row : tensor.getTensor()) {
        for (auto &elt : row) {
            out << elt << " ";
        }
        out << "\n";
    }
    return out;
}