#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/activation_softmax_loss_categoricalcrossentropy.hpp"

class TestActivationSoftmaxLossCategoricalCrossEntropy: public CxxTest::TestSuite {

    public:
        
        void TestActivationSoftmaxLossCategoricalCrossEntropyBackward(void) {
            TS_TRACE("test ActivationSoftmaxLossCategoricalCrossEntropy backward test");
            TensorInline inputs({4, 3});
            inputs.tensor = {0.1, 0.3, 0.6, 0.3, 0.5, 0.2, 0.333333, 0.333333, 0.333333, 0.1, 0.8, 0.1};

            TensorInline y_true({1, 4});
            y_true.tensor = {2.0, 1.0, 0.0, 1.0};

            // One hot encoded
            TensorInline y_true_hot_encoded({4, 3});
            y_true_hot_encoded.tensor = {0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};

            TensorInline expected_output({4, 3});
            expected_output.tensor = {0.025, 0.075, -0.1, 0.075, -0.125, 0.05, -0.16666675, 0.08333325, 0.08333325, 0.025, -0.05, 0.025};
            
            TensorInline dinputs = Activation_Softmax_Loss_CategoricalCrossentropy::backward(inputs, y_true);
            TS_ASSERT((dinputs - expected_output).abs() <= 1e-5);

            dinputs = Activation_Softmax_Loss_CategoricalCrossentropy::backward(inputs, y_true_hot_encoded);
            TS_ASSERT((dinputs - expected_output).abs() <= 1e-5);
        }
};