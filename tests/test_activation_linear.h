#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/activation_linear.hpp"

class TestActivationLinear: public CxxTest::TestSuite {

    public:

        void testActivationLinearForward(void) {
            TS_TRACE("Start activation Linear forward trace");
            TensorInline inputs ({2, 3});
            inputs.tensor = {-2, -1, 0, 1, 2, 3};

            Activation_Linear linear;

            linear.forward(inputs, false);
            TS_ASSERT_EQUALS(linear.getOutput(), inputs);
        }

        void testActivationLinearBackward(void) {
            TS_TRACE("Start activation Linear backward trace");
            TensorInline inputs ({2, 3});
            inputs.tensor = {-2, -1, 0, 1, 2, 3};

            TensorInline dvalues({2, 3});
            dvalues.tensor = {2, 3, 1, 5, -6, 0};

            Activation_Linear linear;

            linear.forward(inputs, false);
            linear.backward(dvalues);

            TS_ASSERT_EQUALS(linear.getOutput(), inputs);
            TS_ASSERT_EQUALS(linear.getDinputs(), dvalues);
        }
};