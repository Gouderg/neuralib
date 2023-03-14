#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/layer_dropout.hpp"

class TestLayerDropout: public CxxTest::TestSuite {

    public:
        void testLayerDropoutForwardBackward(void) {
            TS_TRACE("Starting layer dropout forward and backward test");
            Layer_Dropout dropout(0.1);

            TensorInline inputs({1, 8});
            inputs.tensor = {-1.0, -0.2, 0.0, 0.2, 1.0, 2.0, 3.0, 4.0};

            TensorInline output({1, 8});
            output.tensor = {-1.11111111, -0.22222222, 0, 0.22222222, 1.11111111, 2.22222222, 0, 4.44444444};
        
            dropout.forward(inputs);
            dropout.backward(inputs);

            TS_ASSERT((dropout.getOutput() - output).abs() <= 1e-7);
            TS_ASSERT((dropout.getDinputs() - output).abs() <= 1e-7);
        }
};