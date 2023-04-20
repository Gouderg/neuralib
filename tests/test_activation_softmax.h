#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/activation_softmax.hpp"

class TestActivationSoftmax: public CxxTest::TestSuite {

    public:

        void testActivationSoftmaxForward(void) {
            TS_TRACE("Start activation Softmax forward trace");
            TensorInline inputs ({2, 6});
            inputs.tensor = {-1, -0.5, 0, 0.5, 1, 2, -1, -0.5, 0, 0.5, 1, 2};

            TensorInline expected_output({2, 6});
            expected_output.tensor = {0.02679293, 0.04417407, 0.07283072, 0.12007756, 0.19797443, 0.53815029, 0.02679293, 0.04417407, 0.07283072, 0.12007756, 0.19797443, 0.53815029};

            Activation_Softmax softmax;

            softmax.forward(inputs);
            TS_ASSERT((softmax.getOutput() - expected_output) <= 1e-7);
        }

        void testActivationSoftmaxBackward(void) {
            TS_TRACE("Start activation Softmax backward trace");
            TensorInline inputs ({2, 6});
            inputs.tensor = {-1, -0.5, 0, 0.5, 1, 2, -1, -0.5, 0, 0.5, 1, 2};

            TensorInline expected_output({2, 6});
            expected_output.tensor = {-0.06123346, -0.07886987, -0.09361908, -0.09431297, -0.0565086, 0.38454399, -0.06123346, -0.07886987, -0.09361908, -0.09431297, -0.0565086, 0.38454399};

            Activation_Softmax softmax;

            softmax.forward(inputs);
            softmax.backward(inputs);

            TS_ASSERT((softmax.getDinputs() - expected_output) <= 1e-7);
        }
};