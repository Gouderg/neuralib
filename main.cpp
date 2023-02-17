#include "header/tensor.hpp"
#include "header/tensor_inline.hpp"
#include "header/dataset.hpp"
#include "header/plot.hpp"

#include <chrono>


void test_matrix_product();
void test_addition();
void test_soustraction();
void test_multiplication();
void test_division();
void test_complex();

int main(int argc, char const *argv[]) {

    TensorInline X, y;
    const int samples = 1000;

    X.setHeight(samples * 3);
    X.setWidth(2);
    y.setHeight(1);
    y.setWidth(samples * 3);
    std::tie(X, y) = Dataset::spiral_data(samples, 3);

    
    // Plot the dataset.
    Plot plt;

    plt.set_x_limit(-1, 1);
    plt.set_y_limit(-1, 1);

    for (int i = 0; i < X.getHeight() * X.getWidth(); i += 2) {
        plt.draw_circle(X.tensor[i], X.tensor[i + 1], 0.01 , Plot::getColor(y.tensor[static_cast<int>(i / 2)]));
    }
    plt.show();

    Tensor X1, y1;
    std::tie(X1, y1) = Dataset::spiral_data2(100, 3);

    // Plot the dataset.
    Plot plt2;

    plt2.set_x_limit(-1, 1);
    plt2.set_y_limit(-1, 1);

    for (int i = 0; i < X1.shape().getY(); i++) {
        plt2.draw_circle(X1.getValue(i, 0), X1.getValue(i, 1), 0.01 , Plot::getColor(y1.getValue(0, i)));
    }
    // plt2.show();


    return 0;
}

void test_matrix_product() {
    std::chrono::steady_clock::time_point begin, end, b2, e2;
    const int sizeMat = 4096;

    // Test 1.
   
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

}

void test_addition() {

    TensorInline ti1(3, 3);
    TensorInline ti2(1, 3, 2);
    TensorInline ti3(3, 3);

    ti1.tensor = {2, 2, 2, 2, 2, 2, 2, 2, 2};

    // Tensor +
    std::cout << ti1 + ti2 << std::endl;
    std::cout << ti1 + ti3 << std::endl;

    // Double +
    std::cout << ti1 + 100 << std::endl;

    // Tensor += 
    std::cout << ti1 << std::endl;
    ti1 += ti2;
    std::cout << ti1 << std::endl;

    // Double +=
    ti1 += 2;
    std::cout << ti1 << std::endl;

}

void test_soustraction() {

    TensorInline ti1(3, 3);
    TensorInline ti2(1, 3, 2);
    TensorInline ti3(3, 3);

    ti1.tensor = {2, 2, 2, 2, 2, 2, 2, 2, 2};

    // Tensor -
    std::cout << ti1 - ti2 << std::endl;
    std::cout << ti1 - ti3 << std::endl;

    // Double -
    std::cout << ti1 - 100 << std::endl;

    // Tensor -= 
    std::cout << ti1 << std::endl;
    ti1 -= ti2;
    std::cout << ti1 << std::endl;

    // Double -=
    ti1 -= 2;
    std::cout << ti1 << std::endl;

}

void test_multiplication() {

    TensorInline ti1(3, 3);
    TensorInline ti2(1, 3, 2);
    TensorInline ti3(3, 3);

    ti1.tensor = {2, 2, 2, 2, 2, 2, 2, 2, 2};

    // Tensor *
    std::cout << ti1 * ti2 << std::endl;
    std::cout << ti1 * ti3 << std::endl;

    // Double *
    std::cout << ti1 * 100 << std::endl;

    // Tensor *= 
    std::cout << ti1 << std::endl;
    ti1 *= ti2;
    std::cout << ti1 << std::endl;

    // Double *=
    ti1 *= 2;
    std::cout << ti1 << std::endl;

}

void test_division() {

    TensorInline ti1(3, 3);
    TensorInline ti2(1, 3, 2);
    TensorInline ti3(3, 3);

    ti1.tensor = {2, 2, 2, 2, 2, 2, 2, 2, 2};

    // Tensor *
    std::cout << ti1 / ti2 << std::endl;
    std::cout << ti1 / ti3 << std::endl;

    // Double *
    std::cout << ti1 / 100 << std::endl;

    // Tensor *= 
    std::cout << ti1 << std::endl;
    ti1 /= ti2;
    std::cout << ti1 << std::endl;

    // Double *=
    ti1 /= 2;
    std::cout << ti1 << std::endl;

}

void test_complex() {
    TensorInline ti1(3, 3, 2);

    std::cout << ti1 << std::endl;
    std::cout << ti1.abs() << std::endl;
    std::cout << ti1.sqrt() << std::endl;
    std::cout << TensorInline::sum(ti1) << std::endl;

    std::cout << ti1.transposate() << std::endl;
    
    TensorInline ti2(3, 0, 1);
    std::cout << ti2 << std::endl;
    std::cout << ti2.transposate() << std::endl;


}