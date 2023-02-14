#include "header/tensor.hpp"
#include "header/tensor_inline.hpp"

#include <chrono>


int main(int argc, char const *argv[]) {


    std::chrono::steady_clock::time_point begin, end, b2, e2;
    const int sizeMat = 4096;

    // Test 1.
    begin = std::chrono::steady_clock::now();
    // Tensor t1(sizeMat, sizeMat, 1);
    // Tensor t2(sizeMat, sizeMat, 1);
    // Tensor t3 = t1.dot(t2);
    end = std::chrono::steady_clock::now();
    
    TensorInline ti1(sizeMat, sizeMat, 1);
    TensorInline ti2(sizeMat, sizeMat, 1);
    b2 = std::chrono::steady_clock::now();
    TensorInline ti3 = TensorInline::dot(ti1, ti2);
    e2 = std::chrono::steady_clock::now();
    
    std::cout << "Dimensions: " << sizeMat << std::endl;
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s] vs " << std::chrono::duration_cast<std::chrono::seconds>(e2 - b2).count()
              << "[s]" << std::endl;
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms] vs " << std::chrono::duration_cast<std::chrono::milliseconds>(e2 - b2).count()
              << "[ms]" << std::endl;

    // Test 2.
    // begin = std::chrono::steady_clock::now();
    // Tensor t2(4096, 4096, 1);
    // end = std::chrono::steady_clock::now();

    // b2 = std::chrono::steady_clock::now();
    // TensorInline ti2(4096, 4096, 1);
    // e2 = std::chrono::steady_clock::now();
    // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s] vs " << std::chrono::duration_cast<std::chrono::seconds>(e2 - b2).count()
    //           << "[s]" << std::endl;
    // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms] vs " << std::chrono::duration_cast<std::chrono::milliseconds>(e2 - b2).count()
    //           << "[ms]" << std::endl;
    
    

    

    return 0;
}
