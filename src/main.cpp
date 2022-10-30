#include "../header/tensor.hpp"


int main(int argc, char const *argv[]) {   

    Tensor t1(3, 4);
    std::cout << t1 << std::endl;    

    Tensor t2(3, 4, 1);
    std::cout << t2 << std::endl;    

    Tensor t3(3, 4, 1);
    std::cout << t3 << std::endl;    

    std::cout << t3 * t2 << std::endl;  

    t1 -= t3;
    std::cout << t1 << std::endl;    


    return 0;
}
