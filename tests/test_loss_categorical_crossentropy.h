#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/loss.hpp"

class TestLossCategoricalCrossEntropy: public CxxTest::TestSuite {

    public:
        void testLossCategoricalCrossEntropyForward(void) {
            TS_TRACE("Starting loss categorical crossentropy forward test");
            
            TensorInline inputs({4, 3});
            inputs.tensor = {0.1, 0.3, 0.6, 0.3, 0.5, 0.2, 0.333333, 0.333333, 0.333333, 0.3, 0.8, 0.0};

            TensorInline y_true({1, 4});
            y_true.tensor = {2.0, 1.0, 0.0, 1.0};

            // One hot encoded
            TensorInline y_true_hot_encoded({4, 3});
            y_true_hot_encoded.tensor = {0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};


            double result_expected = 0.6314324110771888;
            Loss_CategoricalCrossEntropy loss;

            TS_ASSERT_EQUALS(loss.calculate(inputs, y_true), result_expected);
            TS_ASSERT_EQUALS(loss.calculate(inputs, y_true_hot_encoded), result_expected);

        }

        void testLossCategoricalCrossEntropyBackward(void) {
            TS_TRACE("Starting loss categorical crossentropy backward test");

            TensorInline inputs({4, 3});
            inputs.tensor = {0.1, 0.3, 0.6, 0.3, 0.5, 0.2, 0.333333, 0.333333, 0.333333, 0.1, 0.8, 0.1};

            TensorInline y_true({1, 4});
            y_true.tensor = {2.0, 1.0, 0.0, 1.0};

            // One hot encoded
            TensorInline y_true_hot_encoded({4, 3});
            y_true_hot_encoded.tensor = {0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};

            TensorInline expected_output({4, 3});
            expected_output.tensor = {0.0, 0.0, -0.4166667, 0.0, -0.5, 0.0, -0.750000, 0.0, 0.0, 0.0, -0.3125, 0.0};
            
            Loss_CategoricalCrossEntropy loss;

            loss.backward(inputs, y_true);
            TS_ASSERT((loss.getDinputs() - expected_output).abs() <= 1e-5);

            loss.backward(inputs, y_true_hot_encoded);
            TS_ASSERT((loss.getDinputs() - expected_output).abs() <= 1e-5);
        }
};