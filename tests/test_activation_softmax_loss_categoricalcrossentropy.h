#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/activation_softmax_loss_categoricalcrossentropy.hpp"

class TestActivationSoftmaxLossCategoricalCrossEntropy: public CxxTest::TestSuite {

    public:
        
        void TestActivationSoftmaxLossCategoricalCrossEntropyForward(void) {
            TS_TRACE("test ActivationSoftmaxLossCategoricalCrossEntropy forward test");
        
        
            TensorInline inputs({4, 3});
            inputs.tensor = {0.1, 0.3, 0.6, 0.3, 0.5, 0.2, 0.333333, 0.333333, 0.333333, 0.1, 0.8, 0.1};

            TensorInline y_true({1, 4});
            y_true.tensor = {2.0, 1.0, 0.0, 1.0};

            // One hot encoded
            TensorInline y_true_hot_encoded({4, 3});
            y_true_hot_encoded.tensor = {0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};


            double result_expected = 0.8953641364530088;
            Activation_Softmax_Loss_CategoricalCrossentropy loss;

            TS_ASSERT(std::abs(loss.forward(inputs, y_true) - result_expected) <= 1e-7);
            TS_ASSERT(std::abs(loss.forward(inputs, y_true_hot_encoded) - result_expected) <= 1e-7);
        }

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
            
            Activation_Softmax_Loss_CategoricalCrossentropy loss;

            loss.backward(inputs, y_true);
            TS_ASSERT((loss.getDinputs() - expected_output).abs() <= 1e-5);

            loss.backward(inputs, y_true_hot_encoded);
            TS_ASSERT((loss.getDinputs() - expected_output).abs() <= 1e-5);
        }
};