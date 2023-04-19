#include "../header/model.hpp"

Model::Model() {
    // Add Layer_Input in layers.
    this->layers.push_back(new Layer_Input());

    // Setup statistic metrics.
    this->stat = new Statistic();
}

void Model::add(Layer* layer) {
    this->layers.push_back(layer);
    if (layer->isTrainable()) {
        this->trainable_layers.push_back(dynamic_cast<Layer_Dense *>(layer));
    }
}

void Model::set(Loss* loss, Optimizer* opti, Accuracy* accuracy) {
    this->loss = loss;
    this->optimizer = opti;
    this->accuracy = accuracy;
}

TensorInline Model::forward(const TensorInline& X, const bool training) {

    int layer_count = static_cast<int>(this->layers.size());

    // Iterate over all layers.
    for (int j = 0; j <= layer_count-1; j++) {
        if (j == 0) {
            this->layers[j]->forward(X, training);
        } else {
            this->layers[j]->forward(this->layers[j-1]->getOutput(), training);
        }
    }

    return this->layers[layer_count - 1]->getOutput();
}

void Model::backward(const TensorInline& output, const TensorInline& y) {

    int layer_count = static_cast<int>(this->layers.size());

    this->loss->backward(output, y);

    for (int j = layer_count-1; j >= 0; j--) {
        if (j == layer_count - 1) {
            this->layers[j]->backward(this->loss->getDinputs());
        } else {
            this->layers[j]->backward(this->layers[j+1]->getDinputs());       
        }
    }
}

void Model::train(ModelParameters p) {

    // Plot the dataset.
    this->plotDatasets(p.plotData, p.data.X, p.data.y);


    // Get the time to compute the execution time.
    time_t start = std::time(NULL);

    // Init accuracy.
    this->accuracy->init(p.data.y, true);

    double loss = 0.0, data_loss = 0.0, regularization_loss = 0.0, accuracy = 0.0;
    double loss_val = 0.0, accuracy_val = 0.0; 
    for (int i = 0; i < p.epochs+1; i++) {

        // Forward.
        TensorInline output = this->forward(p.data.X, true);

        // Loss.
        data_loss = this->loss->calculate(output, p.data.y);
        regularization_loss = 0.0;
        for (auto layer : this->trainable_layers) {
            regularization_loss += this->loss->regularization_loss(*layer);
        }
        loss = data_loss + regularization_loss;

        // Accuracy.
        accuracy = this->accuracy->calculate(output, p.data.y);

        // Backward.
        this->backward(output, p.data.y);

        // Optimization.
        this->optimizer->pre_update_params();
        for (auto layer : this->trainable_layers) {
            this->optimizer->update_params(*layer);
        }
        this->optimizer->post_update_params();

        // Informations.
        if (i % p.print_every == 0) {
            std::cout << "Epoch " << i;
            std::cout << ", acc: " << accuracy;
            std::cout << ", loss: " << loss;
            std::cout << ", (data_loss: " << data_loss;
            std::cout << ", regu_loss: " << regularization_loss;
            std::cout << "), lr: " << this->optimizer->getCurrentLr() << std::endl;
        }
        this->stat->update(loss, accuracy, this->optimizer->getCurrentLr());
    }

    // Validation data.
    TensorInline output_val = this->forward(p.validatation_data.X, false);

    loss_val = this->loss->calculate(output_val, p.validatation_data.y);
    accuracy_val = this->accuracy->calculate(output_val, p.validatation_data.y);
    
    std::cout << "Validation data, acc: " << accuracy_val << ", loss: " << loss_val << std::endl;
    this->plotDatasets(p.plotData, p.validatation_data.X, output_val);

    // Compute execution time.
    time_t end = std::time(NULL);
    std::cout << "Temps d'exécution: " << end - start << " sec." << std::endl;

    // Plot all the stats.
    if (p.printStatistic) {
        this->stat->plot(false);
    }
}

void Model::plotDatasets(PlotConfiguration conf, const TensorInline& X, const TensorInline& y) {
    Plot plt;

    plt.set_x_limit(-1, 1);
    plt.set_y_limit(-1, 1);

    if (conf == PlotConfiguration::line) {
        plt.draw_line(y.tensor, "blue");
    } 
    
    else if (conf == PlotConfiguration::circle) {
        int color = 0;
        for (int i = 0; i < X.getHeight() * X.getWidth(); i += 2) {
            color = TensorInline::round(y.tensor[static_cast<int>(i / 2)]);
            plt.draw_circle(X.tensor[i], X.tensor[i + 1], Plot::getColor(color));
        }
        plt.show();
    }
}