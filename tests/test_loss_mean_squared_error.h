#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/loss.hpp"

class TestLossMeanSquaredError: public CxxTest::TestSuite {

    public:
        void testLossMeanSquaredErrorForward(void) {
            TS_TRACE("Starting Loss_MeanSquaredErrorForward test");

            Loss_MeanSquaredError loss;

            TensorInline y_true({6, 1});
            y_true.tensor = {1, 2, 3, 4, 5, 6};

            TensorInline y_pred({6, 1});
            y_pred.tensor = {-1, -2, -3, -4, -5, -6};

            TensorInline expected_output({1, 6});
            expected_output.tensor = {4, 16, 36, 64, 100, 144};

            TensorInline tested_ouput({1, 6});
            tested_ouput.tensor = loss.forward(y_pred, y_true);
            
            TS_ASSERT_EQUALS(tested_ouput, expected_output);



            TensorInline y_true2({6, 2});
            y_true2.tensor = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};

            TensorInline y_pred2({6, 2});
            y_pred2.tensor = {-1, -2, -3, -4, -5, -6, -1, -2, -3, -4, -5, -6};

            TensorInline expected_output2({1, 6});
            expected_output2.tensor = {10, 50, 122, 10, 50, 122};

            TensorInline tested_ouput2({1, 6});
            tested_ouput2.tensor = loss.forward(y_pred2, y_true2);
            TS_ASSERT_EQUALS(tested_ouput2, expected_output2);

        }

        void testLossMeanSquaredErrorBackward(void) {
            TS_TRACE("Starting Loss_MeanSquaredErrorBackward test");

            Loss_MeanSquaredError loss;

            TensorInline y_true({6, 2});
            y_true.tensor = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};

            TensorInline dvalues({6, 2});
            dvalues.tensor = {-1, -2, -3, -4, -5, -6, -1, -2, -3, -4, -5, -6};

            TensorInline expected_output({6, 2});
            expected_output.tensor = {-0.33333333, -0.66666667, -1.0, -1.33333333, -1.66666667, -2.0, -0.33333333, -0.66666667, -1.0, -1.33333333, -1.66666667, -2.0};

            loss.backward(dvalues, y_true);
            
            TS_ASSERT((loss.getDinputs() - expected_output).abs() <= 1e-7);
        }
};