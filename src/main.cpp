#include "../header/matrix.hpp"


int main(int argc, char const *argv[]) {
    srand(time(NULL));



    std::vector<std::vector<double>> m1 = {{2, 1},
                                           {-1, 0},
                                           {-1, 6}};
                                           
    std::vector<std::vector<double>> m2 = {{3, -2}};
    

    m1 = Matrix::sub(m1, m2);

    // Vérification de matrice.
    for (auto elt : m1) {
        for (auto a : elt) {
            std::cout << a << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
