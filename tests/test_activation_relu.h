#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/activation_relu.hpp"

class TestActivationRelu: public CxxTest::TestSuite {

    public:

        void testActivationReluForward(void) {
            TS_TRACE("Start activation Relu forward trace");
            TensorInline inputs ({2, 3});
            inputs.tensor = {-2, -1, 0, 1, 2, 3};

            TensorInline expected_output({2, 3});
            expected_output.tensor = {0, 0, 0, 1, 2, 3};

            Activation_ReLU relu;

            relu.forward(inputs, false);
            TS_ASSERT_EQUALS(relu.getOutput(), expected_output);
        }

        void testActivationReluBackward(void) {
            TS_TRACE("Start activation Relu backward trace");
            TensorInline inputs ({2, 3});
            inputs.tensor = {-2, -1, 0, 1, 2, 3};

            TensorInline dvalues({2, 3});
            dvalues.tensor = {2, 3, 1, 5, -6, 0};

            TensorInline expected_output({2, 3});
            expected_output.tensor = {0, 0, 0, 5, -6, 0};

            Activation_ReLU relu;

            relu.forward(inputs, false);
            relu.backward(dvalues);

            TS_ASSERT_EQUALS(relu.getDinputs(), expected_output);
        }
};