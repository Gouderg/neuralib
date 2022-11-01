#include "../header/dataset.hpp"

std::tuple<Tensor, Tensor> Dataset::spiral_data(const int samples, const int classes) {
    
    // Normal distribution. 
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{MEAN,DEVIATION};

    Tensor X(samples * classes, 2);
    Tensor y(1, samples * classes);

    double r = 0.0;
    double t = 0.0;
    double step_T = 4.0 / static_cast<double>(samples-1);
    
    for (int i = 0; i < classes; i++) {
        t = i * 4 + d(gen);
        for (int j = 0; j < samples; j++) {
            r = j / static_cast<double>(samples-1);
            X.addValue(i*10 + j, 0, r * sin(t*2.5));
            X.addValue(i*10 + j, 1, r * cos(t*2.5));
            y.addValue(0, i*10 + j, i);
            t += step_T + d(gen);
        }
    }

    return std::make_tuple(X, y);
}