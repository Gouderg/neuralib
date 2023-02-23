#include "../header/statistic.hpp"

void Statistic::update(const double loss, const double accuracy, const double learning_rate) {
    this->loss.push_back(loss);
    this->accuracy.push_back(accuracy);
    this->learning_rate.push_back(learning_rate);
}

void Statistic::plot(const bool wantNormalize) {

    // Normalize data if too much value.
    std::vector<double> loss_ok, acc_ok, lr_ok;
    if (this->loss.size() >= 5000 && wantNormalize) {
        double loss = 0, acc = 0, lr = 0;
        for (int i = 0; i < static_cast<int>(this->loss.size()); i++) {
                loss += this->loss[i];
                acc += this->accuracy[i];
                lr += this->learning_rate[i];
            if (i % 100 == 0) {
                loss_ok.push_back(loss / 100.0);
                acc_ok.push_back(acc / 100.0);
                lr_ok.push_back(lr / 100.0);
                loss = 0;
                acc = 0;
                lr = 0;
            }
        }
    } else {
        loss_ok = this->loss;
        acc_ok = this->accuracy;
        lr_ok = this->learning_rate;
    }

    this->graph.setMultiplot(2, 2);

    // Plot Loss.
    this->graph.set_legend("epoch", "", "Loss");
    this->graph.draw_line(loss_ok, "red");

    // Plot Accuracy.
    this->graph.set_legend("epoch", "", "Accuracy");
    this->graph.draw_line(acc_ok, "blue");

    // Plot learning rate.
    this->graph.set_legend("epoch", "", "Learning rate");
    this->graph.draw_line(lr_ok, "green");

    this->graph.unsetMultiplot();
}