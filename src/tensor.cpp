#include "../header/tensor.hpp"


// Constructor.
Tensor::Tensor(const int nb_col, const int nb_row, const int whichInit) {

    // Gaussian distribution.    
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(MEAN, STD_DEVIATION);
    srand(time(NULL)); 
    switch (whichInit) {

        case 2:
            for (int i = 0; i < nb_col; i++) {
                std::vector<double> v1(nb_row, 1.0);
                this->tensor.push_back(v1);
            }
            break;
            
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
                std::vector<double> v1(nb_row, 0.0);
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
                t3.setValue(i, j, this->tensor[i][j] + t2_tensor[0][j]);
            }
        }
        return t3;
    }
    

    // Other case.
    if (this->tensor.size() != t2_tensor.size() || this->tensor[0].size() != t2_tensor[0].size()) { return *this; }

    for (int i = 0; i < this->tensor.size(); i++) {
        for (int j = 0; j < this->tensor[i].size(); j++) {
            t3.setValue(i, j, this->tensor[i][j] + t2_tensor[i][j]);
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

Tensor Tensor::operator / (double const &n) {
    Tensor t3 = *this;
    if (n == 0) {return t3; }

    for (int i = 0; i < this->tensor.size(); i++) {
        for (int j = 0; j < this->tensor[i].size(); j++) {
            t3.setValue(i, j, this->tensor[i][j] / n);
        }
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

    if (this->tensor.size() != t2_tensor.size() || this->tensor[0].size() != t2_tensor[0].size()) { 
        std::cout << "Wrong dimension for multiplication." << std::endl;
        return *this; 
    }

    for (int i = 0; i < this->tensor.size(); i++) {
        std::vector<double> v1;
        for (int j = 0; j < this->tensor[i].size(); j++) {
            v1.push_back(this->tensor[i][j] * t2_tensor[i][j]);
        }
        t3.addRow(v1);
    }

    return t3;
}

Tensor Tensor::operator * (double const &n) {
    Tensor t3;
    for (int i = 0; i < this->tensor.size(); i++) {
        std::vector<double> v1;
        for (int j = 0; j < this->tensor[i].size(); j++) {
            v1.push_back(this->tensor[i][j] * n);
        }
        t3.addRow(v1);
    }
    return t3;
}


void Tensor::operator *= (Tensor const &t2) {

    std::vector<std::vector<double>> t2_tensor = t2.getTensor();

    if (this->tensor.size() != t2_tensor.size() || this->tensor[0].size() != t2_tensor[0].size()) { 
        std::cout << "Wrong dimension for self multiplication." << std::endl;
        return ; 
    }

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
    if (this->tensor[0].size() != t2_tensor.size()) { 
        std::cout << "Wrong dimension for dot product." << std::endl;
        return *this;
    }

    Tensor output(this->tensor.size(), t2_tensor[0].size());
    double val = 0.0;

    for (int i = 0; i < this->tensor.size(); i++) {
        for (int j = 0; j < t2_tensor[0].size();j++) {
            val = 0.0;
            for (int k = 0; k < this->tensor[0].size(); k++) {
                val += this->tensor[i][k] * t2_tensor[k][j];
            }
            output.setValue(i, j, val);
        }
    }
    return output;
}

Tensor Tensor::dot(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2) {

    Tensor output(v1.size(), v2[0].size());
    double val = 0.0;

    for (int i = 0; i < v1.size(); i++) {
        for (int k = 0; k < v1[0].size(); k++) {
            val = 0.0;
            for (int j = 0; j < v2[0].size();j++) {
                val += v1[i][k] * v2[k][j];
            }
            output.setValue(i, k, val);
        }
    }
    return output;
}

// Inner product between Tensor and 1-D array.
std::vector<double> Tensor::dot(std::vector<double> v1) {

    std::vector<double> output (this->tensor[0].size(), 0);

    for (int i = 0; i < this->tensor[0].size(); i++) {
        for (int j = 0; j < this->tensor.size(); j++) {
            output[i] += this->tensor[i][j] * v1[j];
        }
    }

    return output;
}

Tensor Tensor::sqrt() {
    Tensor output;

    // Run through each column.
    for (int i = 0; i < this->tensor.size(); i++) {
        std::vector<double> line_output;
        // Run through each line.
        for (int j = 0; j < this->tensor[i].size(); j++) {
           line_output.push_back(std::sqrt(this->tensor[i][j]));
        }

        output.addRow(line_output);
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

std::vector<std::vector<double>> Tensor::transposate(std::vector<std::vector<double>> v1) {
    std::vector<std::vector<double>> output;

    // Run through each column.
    for (int j = 0; j < v1[0].size(); j++) {
        std::vector<double> line_output;
        // Run through each line.
        for (int i = 0; i < v1.size(); i++) {
           line_output.push_back(v1[i][j]);
        }
        output.push_back(line_output);
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

// Sum of all the term.
double Tensor::sum(Tensor v1) {
    double sum = 0;


    for (int i = 0; i < v1.shapeY(); i++) {
        std::vector<double> row = v1.getRow(i);
        sum += std::accumulate(row.begin(), row.end(), 0);
    }

    return sum;
}

// Abs of all the term.
Tensor Tensor::abs() {
    Tensor v1 = *this;

    for (int i = 0; i < this->shapeY(); i++){
        for (int j = 0; j < this->shapeX(); j++){
            v1.setValue(i, j, std::abs(this->getValue(i,j)));
        }
    }

    return v1;
}
