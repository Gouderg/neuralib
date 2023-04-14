#include "../header/model.hpp"

Model::Model() {
    this->layers.push_back(new Layer_Input());
}

void Model::add(Layer* layer) {
    this->layers.push_back(layer);
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
        TensorInline output = this->forward(params.X, true);
    }
}

TensorInline Model::forward(const TensorInline& X, const bool training) {

    int layer_count = static_cast<int>(this->layers.size());
    TensorInline output({1, 3});

    std::cout << "hello je suis le forwad" << std::endl;

    return output;

}

void Model::backward(const TensorInline& output, const TensorInline& y) {

    for (auto& elt : this->layers) {
        elt->backward(output);
    }
}

Model::~Model() {
    delete this->accuracy;
    delete this->loss;
    delete this->optimizer;
}