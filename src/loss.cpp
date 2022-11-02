#include "../header/loss.hpp"

double Loss::calculate(Tensor& output, Tensor& y) {

    std::vector<double> samples_losses = forward(output, y);

    return std::reduce(samples_losses.begin(), samples_losses.end()) / samples_losses.size();
}

std::vector<double> Loss::forward(Tensor& y_pred, Tensor& y_true) {
    std::vector<double> a(y_true.shapeY(), 0);
    return a;
}

std::vector<double> Loss_CategoricalCrossEntropy::forward(Tensor& y_pred, Tensor& y_true) {

    // Clip data to prevent division by 0.
    std::vector<std::vector<double>> y_pred_clipped = y_pred.getTensor();

    for (int i = 0; i < y_pred_clipped.size(); i++) {
        for (int j = 0; j < y_pred_clipped[0].size(); j++) {
            if ((y_pred_clipped[i][j] < 1e-15)) {
                y_pred_clipped[i][j] = 1e-15;
            }

            if ((y_pred_clipped[i][j] > 1 - 1e-15)) {
                y_pred_clipped[i][j] = 1 - 1e-15;
            }
        }
    }

    // Probabilities for target values only if categoricals values.
    std::vector<double> correct_confidences(y_true.shapeX(), 0);

    if (y_true.shapeY() == 1) {
        for (int i = 0; i < y_true.shapeX(); i++) {
            correct_confidences[i] = y_pred_clipped[i][y_true.getValue(0, i)];
        }
    } else {
        double somme = 0;
        for (int i = 0; i < y_true.shapeX(); i++) {
            for (int j = 0; j < y_true.shapeY(); j++) {
                correct_confidences[i] += y_pred_clipped[i][j] * y_true.getValue(i, j);
            }
        }
    }

    // Losses.
    for (int i = 0; i < correct_confidences.size(); i++) {
        correct_confidences[i] = -log(correct_confidences[i]);
    }

    return correct_confidences;
}