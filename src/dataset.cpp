#include "../header/dataset.hpp"

std::tuple<TensorInline, TensorInline> Dataset::spiral_data(const int samples, const int classes) {
    
    // Normal distribution. 
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{MEAN, STD_DEVIATION};

    TensorInline X(samples * classes, 2);
    TensorInline y(1, samples * classes);

    double r = 0.0; // Radius for the angle.
    double t = 0.0; // theta
    int color = -1;
    int cpt = 0;
    double step_T = 4.0 / static_cast<double>(samples-1);
    
    for (int i = 0; i < classes * samples * 2; i += 2) {
        if (i % (samples * 2) == 0) {
            color += 1;
            cpt = 0;
            t = color * 4 + (d(gen));
            // std::cout << "\n";
        }

        r = cpt / static_cast<double>(samples-1);
        X.tensor[i] = r * sin(t*2.5);
        X.tensor[i + 1] = r * cos(t*2.5);
        y.tensor[static_cast<int>(i/2)] = color;
        t += (step_T + (d(gen)));
        cpt += 1;
        // std::cout << r << " ";
    }
    std::cout << std::endl;
    return std::make_tuple(X, y);
}

std::tuple<Tensor, Tensor> Dataset::spiral_data2(const int samples, const int classes) {
    
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