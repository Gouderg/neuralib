#include "../header/statistic.hpp"

void Statistic::update(const double loss, const double accuracy) {
    this->loss.push_back(loss);
    this->accuracy.push_back(accuracy);
}

void Statistic::plot() {

    // this->graph.setMultiplot(1, 2);


    // Plot Loss.
    this->graph.set_x_limit(0, this->loss.size());
    this->graph.set_y_limit(0, *max_element(this->loss.begin(), this->loss.end()));
    this->graph.set_legend("epoch", "val", "Loss");

    this->graph.draw_line(this->loss, "blue");

    // Plot Accuracy.
    this->graph.set_x_limit(0, this->accuracy.size());
    this->graph.set_y_limit(0, *max_element(this->accuracy.begin(), this->accuracy.end()));
    this->graph.set_legend("epoch", "val", "Accuracy");

    this->graph.draw_line(this->accuracy, "green");


    // Plot learning rate.
    // this->graph.unsetMultiplot();

}