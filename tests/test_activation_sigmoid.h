#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/activation_sigmoid.hpp"

class TestActivationSigmoid: public CxxTest::TestSuite {

    public:

        void testActivationSigmoidForward(void) {
            TS_TRACE("Start activation Sigmoid forward trace");
            TensorInline inputs ({1, 3});
            inputs.tensor = {-2000, 0, 3000};

            TensorInline expected_output({1, 3});
            expected_output.tensor = {0, 0.5, 1,};

            Activation_Sigmoid sigmoid;

            sigmoid.forward(inputs, false);

            TS_ASSERT_EQUALS(sigmoid.getOutput(), expected_output);
        }

        void testActivationSigmoidBackward(void) {
            TS_TRACE("Start activation Sigmoid backward trace");
            TensorInline inputs ({1, 4});
            inputs.tensor = {-2000.0, 0.0, 1000.0, 0.0};

            TensorInline dvalues({1, 4});
            dvalues.tensor = {0.0, 0.5, 1.0, 50.0};

            TensorInline expected_output({1, 4});
            expected_output.tensor = {0, 0.125, 0.0, 12.5};

            Activation_Sigmoid sigmoid;

            sigmoid.forward(inputs);
            sigmoid.backward(dvalues);

            TS_ASSERT_EQUALS(sigmoid.getDinputs(), expected_output);
        }
};