#include "../header/activation_softmax_loss_categoricalcrossentropy.hpp"

TensorInline Activation_Softmax_Loss_CategoricalCrossentropy::backward(const TensorInline &dvalues, const TensorInline& y_true) {

    TensorInline dinputs = dvalues;

    // if labels are one-hot encoded.
    std::vector<double> y_flat (y_true.getHeight() == 1 ? y_true.getWidth() : y_true.getHeight(), 0.0);
    

    if (y_true.getHeight() == 1) {
        y_flat = y_true.tensor;
    } else {
        int w = y_true.getWidth();
        for (int i = 0; i < y_true.getHeight(); i++) {
            for (int j = 0; j < w; j++) {
                if (y_true.tensor[i * w + j]) {
                    y_flat[i] = j;
                }
            }
        }
    }

    // Compute the gradient (-1 on the good label).
    double samples = y_flat.size();
    for (int i = 0; i < samples; i++) {
        dinputs.tensor[i * dinputs.getWidth() + y_flat[i]] -= 1.0;
    }
    return dinputs / samples;
}