#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/loss.hpp"

class TestLossBinaryCrossEntropy: public CxxTest::TestSuite {

    public:
        void testLossBinaryCrossEntropyForward(void) {
            TS_TRACE("Starting Loss_BinaryCrossentropyForward test");

            Loss_BinaryCrossentropy loss;

            TensorInline y_true({6, 1});
            y_true.tensor = {0, 0, 0, 1, 1, 1};

            TensorInline y_pred({6, 1});
            y_pred.tensor = {0.5, 0.4999578, 0.50000936, 0.5, 0.500017, 0.49976814};

            TensorInline expected_output({1, 6});
            expected_output.tensor = {0.6931472, 0.6930628, 0.6931659, 0.6931472, 0.6931132, 0.693611};

            TensorInline tested_ouput({1, 6});
            tested_ouput.tensor = loss.forward(y_pred, y_true);
            
            TS_ASSERT((tested_ouput - expected_output).abs() <= 1e-7);
        }

        void testLossBinaryCrossEntropyBackward(void) {
            TS_TRACE("Starting Loss_BinaryCrossentropyBackward test");

            Loss_BinaryCrossentropy loss;

            TensorInline y_true({6, 1});
            y_true.tensor = {0, 0, 0, 1, 1, 1};

            TensorInline dvalues({6, 1});
            dvalues.tensor = {0.5, 0.4999578, 0.50000936, 0.5, 0.500017, 0.49976814};

            TensorInline expected_output({6, 1});
            expected_output.tensor = {0.33333334, 0.3333052, 0.33333957, -0.33333334, -0.33332202, -0.333488};

            loss.backward(dvalues, y_true);
            
            TensorInline tested_ouput = loss.getDinputs();

            TS_ASSERT((tested_ouput - expected_output).abs() <= 1e-7);
        }
};