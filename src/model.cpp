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

void Model::train(ModelOptions p) {

    // Plot the dataset.
    this->plotDatasets(p.plotData, p.data.X, p.data.y);

    // Get the time to compute the execution time.
    time_t start = std::time(NULL);

    // Init accuracy.
    this->accuracy->init(p.data.y, true);

    // Calculate number of step.
    int train_steps = p.data.X.getHeight() / p.batch_size;
    int validation_steps = p.validatation_data.X.getHeight() / p.batch_size;

    // To get the full dataset.
    if (p.batch_size * train_steps < p.data.X.getHeight()) {
        train_steps += 1;
    }

    if (p.batch_size * train_steps < p.validatation_data.X.getHeight()) {
        validation_steps += 1;
    }

    
    TensorInline batch_X({p.batch_size, p.data.X.getWidth()});
    TensorInline batch_y({1, p.batch_size});

    double loss = 0.0, accuracy = 0.0, epoch_loss = 0.0, epoch_accuracy = 0.0, validation_loss = 0.0, validation_accuracy = 0.0;
    for (int epoch = 0; epoch < p.epochs+1; epoch++) {

        // Epochs.
        std::cout << "epoch: " << epoch << std::endl;

        // Reset accumulated value in loss and accuracy.
        this->loss->new_pass();
        this->accuracy->new_pass();

        batch_X.setHeight(p.batch_size);
        batch_y.setWidth(p.batch_size);
        
        for (int step = 0; step < train_steps; step++) {

            if ((step+1) * p.batch_size > p.data.X.getHeight()) {
                batch_X.tensor.assign(p.data.X.tensor.begin() + p.data.X.getWidth() * p.batch_size * step, p.data.X.tensor.end());
                batch_y.tensor.assign(p.data.y.tensor.begin() + p.batch_size * step, p.data.y.tensor.end());
                batch_X.setHeight((step+1) * p.batch_size - p.data.X.getHeight());
                batch_y.setWidth((step+1) * p.batch_size - p.data.y.getWidth());

            } else {
                batch_X.tensor.assign(p.data.X.tensor.begin() + p.data.X.getWidth() * p.batch_size * step, p.data.X.tensor.begin() + p.data.X.getWidth() * p.batch_size * (step + 1));
                batch_y.tensor.assign(p.data.y.tensor.begin() + p.batch_size * step, p.data.y.tensor.begin() + p.batch_size * (step + 1));
            }

            // Forward.
            TensorInline output = this->forward(batch_X, true);

            // Loss.
            LossValues lv = this->loss->calculate(output, batch_y, true);
            loss = lv.data_loss + lv.regularization_loss;

            // Accuracy.
            accuracy = this->accuracy->calculate(output, batch_y);

            // Backward.
            this->backward(output, batch_y);

            // Optimization.
            this->optimizer->pre_update_params();
            for (auto layer : this->trainable_layers) {
                this->optimizer->update_params(*layer);
            }
            this->optimizer->post_update_params();

            // Informations.
            if (step % p.print_every == 0 || step == train_steps - 1) {
                std::cout << "step: " << step;
                std::cout << ", acc: " << accuracy;
                std::cout << ", loss: " << loss;
                std::cout << ", (data_loss: " << lv.data_loss;
                std::cout << ", regu_loss: " << lv.regularization_loss;
                std::cout << "), lr: " << this->optimizer->getCurrentLr() << std::endl;
            }
            
            this->stat->update(loss, accuracy, this->optimizer->getCurrentLr());
        }

        // Get and print epoch loss and accuracy
        LossValues elv = this->loss->calculate_accumulated(true);
        epoch_loss = elv.data_loss + elv.regularization_loss;
        epoch_accuracy = this->accuracy->calculate_accumulated();

        // Informations.
        std::cout << ", acc: " << epoch_accuracy;
        std::cout << ", loss: " << epoch_loss;
        std::cout << ", (data_loss: " << elv.data_loss;
        std::cout << ", regu_loss: " << elv.regularization_loss;
        std::cout << "), lr: " << this->optimizer->getCurrentLr() << std::endl;


        // Reset accumulated value in loss and accuracy.
        this->loss->new_pass();
        this->accuracy->new_pass();
        batch_X.setHeight(p.batch_size);
        batch_y.setWidth(p.batch_size);
        for (int step = 0; step < validation_steps; step++) {
            if ((step+1) * p.batch_size > p.validatation_data.X.getHeight()) {
                batch_X.tensor.assign(p.validatation_data.X.tensor.begin() + p.validatation_data.X.getWidth() * p.batch_size * step, p.validatation_data.X.tensor.end());
                batch_y.tensor.assign(p.validatation_data.y.tensor.begin() + p.batch_size * step, p.validatation_data.y.tensor.end());
                batch_X.setHeight((step+1) * p.batch_size - p.data.X.getHeight());
                batch_y.setWidth((step+1) * p.batch_size - p.data.y.getWidth());

            } else {
                batch_X.tensor.assign(p.validatation_data.X.tensor.begin() + p.validatation_data.X.getWidth() * p.batch_size * step, p.validatation_data.X.tensor.begin() + p.validatation_data.X.getWidth() * p.batch_size * (step + 1));
                batch_y.tensor.assign(p.validatation_data.y.tensor.begin() + p.batch_size * step, p.validatation_data.y.tensor.begin() + p.batch_size * (step + 1));
            }
            // Validation data.
            TensorInline output_val = this->forward(batch_X, false);

            LossValues lv_val = this->loss->calculate(output_val, batch_y, false);
            accuracy = this->accuracy->calculate(output_val, batch_y);

        }
        validation_accuracy = this->accuracy->calculate_accumulated();
        LossValues lv_val = this->loss->calculate_accumulated();
        validation_loss = lv_val.data_loss + lv_val.regularization_loss;
        std::cout << "Validation data, acc: " << validation_accuracy << ", loss: " << validation_loss << std::endl;
    }


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
            // If we have 3 value like this 0.01 0.01 0.98, we take the index of the max.
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