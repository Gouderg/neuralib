/**
 * @file matrix.cpp
 * @author Victor ILLIEN (https://github.com/Gouderg)
 * @brief Functions from the Matrix Class.
 * @version 0.1
 * @date 2022-04-20
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "../header/matrix.hpp"

/**
 * Return the matrix product between two vectors.
 * 
 * @param std::vector<std::vector<double>>-m1 First matrix.
 * @param std::vector<std::vector<double>>-m2 Second matrix.
 * @returns Return the matrix product.
 * 
*/
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


/**
 * Return the product between matrix and constant.
 * 
 * @param std::vector<std::vector<double>>-m1 First matrix.
 * @param double n.
 * @returns Return the matrix product.
 * 
*/
std::vector<std::vector<double>> Matrix::product(std::vector<std::vector<double>> m, double n) {
    
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[0].size(); j++) {
            m[i][j] *= n;
        }
    }
    
    return m;
}


/**
 * Return the addition between two vectors.
 * 
 * @param std::vector<std::vector<double>>-m1 First matrix.
 * @param std::vector<std::vector<double>>-m2 Second matrix.
 * @returns Return the result of the addition.
 * 
*/
std::vector<std::vector<double>> Matrix::add(std::vector<std::vector<double>> m1, std::vector<std::vector<double>> m2) {
    
    // Init the return matrix.
    std::vector<std::vector<double>> retour(m1.size(), std::vector<double> (m2[0].size(), 0));

    // Broadcasting addition.
    if (m2.size() == 1 && m1[0].size() == m2[0].size()) {
        for (int i = 0; i < m1.size(); i++) {
            for (int j = 0; j < m2[0].size(); j++) {
                retour[i][j] = m1[i][j] + m2[0][j];
            }
        }
        return retour;
    }
    
    // Check if it is possible. 
    else if (m1.size() != m2.size() || m1[0].size() != m2[0].size()) {
        std::cout << "Impossible de calculer l'addition des matrices" << std::endl;
        return m1;
    }

    // Perform addition.
    for (int i = 0; i < m1.size(); i++) {
        for (int j = 0; j < m2[0].size(); j++) {
            retour[i][j] = m1[i][j] + m2[i][j];
        }
    }

    return retour;
}


/**
 * Return the addition between matrix and constant.
 * 
 * @param std::vector<std::vector<double>>-m1 First matrix.
 * @param double n.
 * @returns Return the result of the addition.
 * 
*/
std::vector<std::vector<double>> Matrix::add(std::vector<std::vector<double>> m, double n) {

    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[0].size(); j++) {
            m[i][j] += n;
        }
    }
    
    return m;
}


/**
 * Return the matrix substraction between two vectors.
 * 
 * @param std::vector<std::vector<double>>-m1 First matrix.
 * @param std::vector<std::vector<double>>-m2 Second matrix.
 * @returns Return the result of the substraction.
 * 
*/
std::vector<std::vector<double>> Matrix::sub(std::vector<std::vector<double>> m1, std::vector<std::vector<double>> m2) {
    
    // Init the return matrix.
    std::vector<std::vector<double>> retour(m1.size(), std::vector<double> (m2[0].size(), 0));

    // Broadcasting addition.
    if (m2.size() == 1 && m1[0].size() == m2[0].size()) {
        for (int i = 0; i < m1.size(); i++) {
            for (int j = 0; j < m2[0].size(); j++) {
                retour[i][j] = m1[i][j] - m2[0][j];
            }
        }
        return retour;
    }
    
    // Check if it is possible. 
    else if (m1.size() != m2.size() || m1[0].size() != m2[0].size()) {
        std::cout << "Impossible de calculer l'addition des matrices" << std::endl;
        return m1;
    }

    // Perform substration.
    for (int i = 0; i < m1.size(); i++) {
        for (int j = 0; j < m2[0].size(); j++) {
            retour[i][j] = m1[i][j] - m2[i][j];
        }
    }

    return retour;
}


/**
 * Return the substraction between matrix and constant.
 * 
 * @param std::vector<std::vector<double>>-m1 First matrix.
 * @param double n.
 * @returns Return the result of the substraction.
 * 
*/
std::vector<std::vector<double>> Matrix::sub(std::vector<std::vector<double>> m, double n){
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[0].size(); j++) {
            m[i][j] -= n;
        }
    }
    return m;
}


/**
 * Return the transpose of the matrix.
 * 
 * @param std::vector<std::vector<double>>-m1 First matrix.
 * @returns Return the matrix.
 * 
*/
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
