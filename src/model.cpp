#include "../header/model.hpp"

Model::Model() {
    // Add Layer_Input in layers.
    this->layers.push_back(new Layer_Input());

    // Setup diagramm class.
    this->plt = new Plot();
    this->plt->set_x_limit(-1, 1);
    this->plt->set_y_limit(-1, 1);

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

    TensorInline X({NB_REGRESSION_POINT, 1}), y({NB_REGRESSION_POINT, 1});
    std::tie(X, y) = p.data; 

    // Plot the dataset.
    if (p.plotData) {
        this->plt->draw_line(y.tensor, "red");
    }

    // Get the time to compute the execution time.
    time_t start = std::time(NULL);

    // Init accuracy.
    this->accuracy->init(y, true);

    double loss = 0.0, data_loss = 0.0, regularization_loss = 0.0, accuracy = 0.0;
    double loss_val = 0.0, accuracy_val = 0.0; 
    for (int i = 0; i < p.epochs+1; i++) {

        // Forward.
        TensorInline output = this->forward(X, true);

        // Loss.
        data_loss = this->loss->calculate(output, y);
        regularization_loss = 0.0;
        for (auto layer : this->trainable_layers) {
            regularization_loss += this->loss->regularization_loss(*layer);
        }
        loss = data_loss + regularization_loss;

        // Accuracy.
        accuracy = this->accuracy->calculate(output, y);

        // Backward.
        this->backward(output, y);

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
    TensorInline X_val({NB_REGRESSION_POINT, 1}), y_val({NB_REGRESSION_POINT, 1});
    std::tie(X_val, y_val) = p.data;

    TensorInline output_val = this->forward(X_val, false);

    loss_val = this->loss->calculate(output_val, y_val);
    accuracy_val = this->accuracy->calculate(output_val, y_val);
    
    std::cout << "Validation data, acc: " << accuracy_val << ", loss: " << loss_val << std::endl;
    if (p.plotData) {
        this->plt->draw_line(output_val.tensor, "blue");
    }

    // Compute execution time.
    time_t end = std::time(NULL);
    std::cout << "Temps d'exécution: " << end - start << " sec." << std::endl;

    // Plot all the stats.
    if (p.printStatistic) {
        this->stat->plot(false);
    }
}
