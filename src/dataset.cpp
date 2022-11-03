#include "../header/dataset.hpp"

std::tuple<Tensor, Tensor> Dataset::spiral_data(const int samples, const int classes) {
    
    // Normal distribution. 
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{MEAN, 1};

    Tensor X(samples * classes, 2);
    Tensor y(1, samples * classes);

    double r = 0.0;
    double t = 0.0;
    double step_T = 4.0 / static_cast<double>(samples-1);
    
    for (int i = 0; i < classes; i++) {
        t = i * 4 + (d(gen) * 0.1);
        for (int j = 0; j < samples; j++) {
            r = j / static_cast<double>(samples-1);
            X.setValue(i*samples + j, 0, r * sin(t*2.5));
            X.setValue(i*samples + j, 1, r * cos(t*2.5));
            y.setValue(0, i*samples + j, i);
            t += step_T + (d(gen) * 0.1);
        }
    }

    return std::make_tuple(X, y);
}