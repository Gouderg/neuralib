#include "../header/matrix.hpp"

std::vector<std::vector<double>> Matrix::product(std::vector<std::vector<double>> m1, std::vector<std::vector<double>> m2) {

    // Check if it is possible. 
    if (m1[0].size() != m2.size()) {
        std::cout << "Impossible de calculer le produit des matrices" << std::endl;
        return m1;
    }
    
    // Init the return matrix.
    std::vector<std::vector<double>> retour(m1.size(), std::vector<double> (m2[0].size(), 0));

    for (int i = 0; i < m1.size(); i++) {
        for (int j = 0; j < m2[0].size();j++) {
            for (int k = 0; k < m1[0].size(); k++) {
                retour[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }

    return retour;
}

std::vector<std::vector<double>> Matrix::product(std::vector<std::vector<double>> m, double n) {
    
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[0].size(); j++) {
            m[i][j] *= n;
        }
    }
    
    return m;
}



std::vector<std::vector<double>> Matrix::add(std::vector<std::vector<double>> m1, std::vector<std::vector<double>> m2) {
    
    // Check if it is possible. 
    if (m1.size() != m2.size() || m1[0].size() != m2[0].size()) {
        std::cout << "Impossible de calculer l'addition des matrices" << std::endl;
        return m1;
    }

    // Init the return matrix.
    std::vector<std::vector<double>> retour(m1.size(), std::vector<double> (m2[0].size(), 0));

    for (int i = 0; i < m1.size(); i++) {
        for (int j = 0; j < m2[0].size(); j++) {
            retour[i][j] = m1[i][j] + m2[i][j];
        }
    }

    return retour;
}

std::vector<std::vector<double>> Matrix::add(std::vector<std::vector<double>> m, double n) {

    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[0].size(); j++) {
            m[i][j] += n;
        }
    }
    
    return m;
}

std::vector<std::vector<double>> Matrix::sub(std::vector<std::vector<double>> m1, std::vector<std::vector<double>> m2) {
    
    // Check if it is possible. 
    if (m1.size() != m2.size() || m1[0].size() != m2[0].size()) {
        std::cout << "Impossible de calculer l'addition des matrices" << std::endl;
        return m1;
    }

    // Init the return matrix.
    std::vector<std::vector<double>> retour(m1.size(), std::vector<double> (m2[0].size(), 0));

    for (int i = 0; i < m1.size(); i++) {
        for (int j = 0; j < m2[0].size(); j++) {
            retour[i][j] = m1[i][j] - m2[i][j];
        }
    }

    return retour;
}

std::vector<std::vector<double>> Matrix::sub(std::vector<std::vector<double>> m, double n) {
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[0].size(); j++) {
            m[i][j] -= n;
        }
    }
    return m;
}

std::vector<std::vector<double>> Matrix::transposate(std::vector<std::vector<double>> m) {

    std::vector<std::vector<double>> output;
    
    // Run through each column.
    for (int j = 0; j < m[0].size(); j++) {
        std::vector<double> line_output;
        
        // Run through each line.
        for (int i = 0; i < m.size(); i++) {
           line_output.push_back(m[i][j]);
        }

        output.push_back(line_output);
    }
    
    return output;
}
