#include "../header/model.hpp"

Model::Model() {
    this->layers.push_back(new Layer_Input());
}

void Model::add(Layer* layer) {
    this->layers.push_back(layer);
    // TODO mettre dans truc qui se souvient
}

void Model::set(Loss* loss, Optimizer* opti, Accuracy* accuracy) {
    this->loss = loss;
    this->optimizer = opti;
    this->accuracy = accuracy;
}

void Model::train(ModelParameters params) {
    this->accuracy->init(params.y);


    double data_loss = 0.0, regularization_loss = 0.0;
    double loss = 0.0;
    for (int i = 1; i < params.epochs + 1; i++) {

        // Forward.
        TensorInline output = this->forward(params.X, true);

        // Loss.
        data_loss = this->loss->calculate(output, params.y);
        regularization_loss = 0.0;
        for (int j = 0; j <= this->layers.size(); j++) {
            if (this->layers[j]->isTrainable()) {
                // regularization_loss += this->loss->regularization_loss(*this->layers[j]);
            }
        }
        loss = data_loss + regularization_loss;

        // Backward.
        this->backward(output, params.y);

        // Optimization.
        this->optimizer->pre_update_params();
        for (int j = 0; j <= this->layers.size(); j++) {
            if (this->layers[j]->isTrainable()) {
                // this->optimizer->update_params(*this->layers[j]);
            }
        }
        this->optimizer->post_update_params();

        // Informations.
        if (i % params.print_every == 0) {
            std::cout << "Epoch " << i;
            std::cout << ", acc: " << accuracy;
            std::cout << ", loss: " << loss;
            std::cout << ", (data_loss: " << data_loss;
            std::cout << ", regu_loss: " << regularization_loss;
            std::cout << "), lr: " << this->optimizer->getCurrentLr() << std::endl;
        }

    }
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
        std::cout << j << ", " << j + 1 << std::endl;
        if (j == layer_count - 1) {
            this->layers[j]->backward(this->loss->getDinputs());
        } else {
            this->layers[j]->backward(this->layers[j+1]->getDinputs());
        }
    }
}

Model::~Model() {
    delete this->accuracy;
    delete this->loss;
    delete this->optimizer;
}