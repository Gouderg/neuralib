#include "../header/dataset.hpp"

std::tuple<TensorInline, TensorInline> Dataset::spiral_data(const int samples, const int classes) {
    
    // Normal distribution. 
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{MEAN, STD_DEVIATION};


    TensorInline X({samples * classes, 2});
    TensorInline y({1, samples * classes});

    double r = 0.0; // Radius for the angle.
    double t = 0.0, t_random = 0.0; // theta
    int color = -1;
    int cpt = 0;
    double step_T = 4.0 / static_cast<double>(samples-1);
    
    for (int i = 0; i < classes * samples * 2; i += 2) {

        if (i % (samples * 2) == 0) {
            color += 1;
            cpt = 0;
            t = color * 4.0; 
        }

        t_random = t + (d(gen)) * 0.2;
        r = cpt / static_cast<double>(samples-1);
        X.tensor[i] = r * sin(t_random*2.5);
        X.tensor[i + 1] = r * cos(t_random*2.5);
        y.tensor[static_cast<int>(i/2)] = color;
        t += step_T;
        cpt += 1;
    }
    return std::make_tuple(X, y);
}