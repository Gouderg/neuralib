#include "../header/statistic.hpp"

void Statistic::update(const double loss, const double accuracy, const double learning_rate) {
    this->loss.push_back(loss);
    this->accuracy.push_back(accuracy);
    this->learning_rate.push_back(learning_rate);
}

void Statistic::plot() {

    this->graph.setMultiplot(2, 2);

    // Plot Loss.
    this->graph.set_legend("epoch", "", "Loss");
    this->graph.draw_line(this->loss, "red");

    // Plot Accuracy.
    this->graph.set_legend("epoch", "", "Accuracy");
    this->graph.draw_line(this->accuracy, "blue");

    // Plot learning rate.
    this->graph.set_legend("epoch", "", "Learning rate");
    this->graph.draw_line(this->learning_rate, "green");

    this->graph.unsetMultiplot();
}