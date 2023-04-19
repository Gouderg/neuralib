#include "../header/model.hpp"

Model::Model() {
    // Add Layer_Input in layers.
    this->layers.push_back(new Layer_Input());

    // Setup statistic metrics.
    this->stat = new Statistic();
}

void Model::add(Layer* layer) {
    this->layers.push_back(layer);
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

    if (this->isSoftmaxClassifierOuput) {
        TensorInline dinputs = Activation_Softmax_Loss_CategoricalCrossentropy::backward(output, y);

        this->layers[layer_count - 1]->setDinputs(dinputs);

        for (int j = layer_count-2; j >= 0; j--) {
            this->layers[j]->backward(this->layers[j+1]->getDinputs());       
        }

        return;
    }

    this->loss->backward(output, y);

    for (int j = layer_count-1; j >= 0; j--) {
        if (j == layer_count - 1) {
            this->layers[j]->backward(this->loss->getDinputs());
        } else {
            this->layers[j]->backward(this->layers[j+1]->getDinputs());       
        }
    }
}

void Model::finalize() {

    // Find all trainable layer.
    for (auto layer : this->layers) {
        if (layer->isTrainable()) {
            this->trainable_layers.push_back(dynamic_cast<Layer_Dense *>(layer));
        }
    }

    // Set the trainable layers in the loss.
    this->loss->setTrainableLayer(this->trainable_layers);

    // Verify if the last layer is Softmax and the loss is CategoricalCrossEntropy.
    if (nullptr != dynamic_cast<Activation_Softmax*>(this->layers[this->layers.size() - 1]) && nullptr != dynamic_cast<Loss_CategoricalCrossEntropy*>(this->loss)) {
        this->isSoftmaxClassifierOuput = true;
    }
}

void Model::train(ModelParameters p) {

    // Plot the dataset.
    this->plotDatasets(p.plotData, p.data.X, p.data.y);

    // Get the time to compute the execution time.
    time_t start = std::time(NULL);

    // Init accuracy.
    this->accuracy->init(p.data.y, true);

    double loss = 0.0, accuracy = 0.0;
    for (int i = 0; i < p.epochs+1; i++) {

        // Forward.
        TensorInline output = this->forward(p.data.X, true);

        // Loss.
        LossValues lv = this->loss->calculate(output, p.data.y, true);
        loss = lv.data_loss + lv.regularization_loss;

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
            std::cout << ", (data_loss: " << lv.data_loss;
            std::cout << ", regu_loss: " << lv.regularization_loss;
            std::cout << "), lr: " << this->optimizer->getCurrentLr() << std::endl;
        }
        this->stat->update(loss, accuracy, this->optimizer->getCurrentLr());
    }

    // Validation data.
    TensorInline output_val = this->forward(p.validatation_data.X, false);

    LossValues lv_val = this->loss->calculate(output_val, p.validatation_data.y, false);
    double accuracy_val = this->accuracy->calculate(output_val, p.validatation_data.y);
    
    std::cout << "Validation data, acc: " << accuracy_val << ", loss: " << lv_val.data_loss << std::endl;
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

        if (y.getHeight() == 1 || y.getHeight() != 1 && y.getWidth() == 1) {
            for (int i = 0; i < X.getHeight() * X.getWidth(); i += NB_INPUTS) {
                color = TensorInline::round(y.tensor[static_cast<int>(i / 2)]);
                plt.draw_circle(X.tensor[i], X.tensor[i + 1], Plot::getColor(color));
            }            
        } else {
                // If we have 3 value like this 0.01 0.01 0.98, we take the index of the max
            int y_index = 0;
            int y_w = y.getWidth();
            for (int i = 0; i < X.getHeight() * X.getWidth(); i += NB_INPUTS) {
                color = std::max_element(y.tensor.begin() + y_index , y.tensor.begin() + y_index + y_w) - (y.tensor.begin() + y_index);
                plt.draw_circle(X.tensor[i], X.tensor[i + 1], Plot::getColor(color));
                y_index += y_w;
            }
        }
        plt.show();
    }
}