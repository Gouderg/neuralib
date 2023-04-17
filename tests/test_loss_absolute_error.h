#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/loss.hpp"

class TestLossMeanAbsoluteError: public CxxTest::TestSuite {

    public:
        void testLossMeanAbsoluteErrorForward(void) {
            TS_TRACE("Starting Loss_MeanAbsoluteErrorForward test");

            Loss_MeanAbsoluteError loss;

            TensorInline y_true({6, 1});
            y_true.tensor = {1, 2, 3, 4, 5, 6};

            TensorInline y_pred({6, 1});
            y_pred.tensor = {-1, -2, -3, -4, -5, -6};

            TensorInline expected_output({1, 6});
            expected_output.tensor = {2, 4, 6, 8, 10, 12};

            TensorInline tested_ouput({1, 6});
            tested_ouput.tensor = loss.forward(y_pred, y_true);
            
            TS_ASSERT_EQUALS(tested_ouput, expected_output);



            TensorInline y_true2({6, 2});
            y_true2.tensor = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};

            TensorInline y_pred2({6, 2});
            y_pred2.tensor = {-1, -2, -3, -4, -5, -6, -1, -2, -3, -4, -5, -6};

            TensorInline expected_output2({1, 6});
            expected_output2.tensor = {3, 7, 11, 3, 7, 11};

            TensorInline tested_ouput2({1, 6});
            tested_ouput2.tensor = loss.forward(y_pred2, y_true2);
            TS_ASSERT_EQUALS(tested_ouput2, expected_output2);

        }

        void testLossMeanAbsoluteErrorBackward(void) {
            TS_TRACE("Starting Loss_MeanAbsoluteErrorBackward test");

            Loss_MeanAbsoluteError loss;

            TensorInline y_true({6, 2});
            y_true.tensor = {-1, 2, 3, 4, -5, 6, 1, -2, 3, 4, 5, 6};

            TensorInline dvalues({6, 2});
            dvalues.tensor = {5, 2, -3, -4, -5, -6, -1, -2, 3, -4, -5, -6};

            TensorInline expected_output({6, 2});
            expected_output.tensor = {-0.08333333, 0, 0.08333333, 0.08333333, 0, 0.08333333, 0.08333333, 0, 0, 0.08333333, 0.08333333, 0.08333333};

            loss.backward(dvalues, y_true);
            
            TS_ASSERT((loss.getDinputs() - expected_output).abs() <= 1e-7);
        }
};