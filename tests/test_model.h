#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/model.hpp"
#include "../header/dataset.hpp"

class TestModel: public CxxTest::TestSuite {

    public:

        void testModelRegression(void) {
            TS_TRACE("Start model regression trace");

            // Init the dataset.
            TensorInline X({1000, 1}), y({1000, 1});
            TensorInline X_test({1000, 1}), y_test({1000, 1});

            // Get the dataset.
            std::tie(X, y) = Dataset::sine_data(1000);
            std::tie(X_test, y_test) = Dataset::sine_data(1000);

            // Create the model.
            Model model;

            // Add all layers.
            model.add(new Layer_Dense({1, NB_NEURON}));
            model.add(new Activation_ReLU());
            model.add(new Layer_Dense({NB_NEURON, NB_NEURON}));
            model.add(new Activation_ReLU());
            model.add(new Layer_Dense({NB_NEURON, 1}));
            model.add(new Activation_Linear());           

            // Setup loss, optimizer and accuracy
            Loss_MeanSquaredError loss_function;
            Optimizer_Adam optimizer = Optimizer_Adam(0.005, 1e-3, MOMENTUM_EPSILON);
            Accuracy_Regression accuracy(STRICT_ACCURACY_METRICS);


            model.set(&loss_function, &optimizer, &accuracy);

            model.train({X, y, std::make_tuple(X_test, y_test), 2});

        }
};