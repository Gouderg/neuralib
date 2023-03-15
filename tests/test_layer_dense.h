#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/layer_dense.hpp"

class TestLayerDense: public CxxTest::TestSuite {

    public:
        void testLayerDenseForward(void) {
            TS_TRACE("Starting layer dense forward test");

            Layer_Dense dense1(6, 2, 3.0);
            TensorInline weights({6, 2});
            weights.tensor = { 0.17640524, 0.04001572, 0.0978738, 0.22408931, 0.1867558, -0.09772779, 0.09500884, -0.01513572, -0.01032189, 0.04105985, 0.01440436, 0.14542735};

            dense1.setWeights(weights);

            TensorInline inputs({2, 6});
            inputs.tensor = {-1, -0.5, 0, 0.5, 1, 2, -1, -0.5, 0, 0.5, 1, 2};

            TensorInline output({2, 2});
            output.tensor = {-0.15935089, 0.1722863, -0.15935089, 0.1722863};

            dense1.forward(inputs);

            TS_ASSERT_EQUALS(dense1.getWeightRegL1(), 3.0);
            TS_ASSERT_EQUALS(dense1.getBiasRegL1(), 0.0);
            TS_ASSERT_EQUALS(dense1.getWeightRegL2(), 0.0);
            TS_ASSERT_EQUALS(dense1.getBiasRegL2(), 0.0);
            TS_ASSERT((dense1.getOutput() - output).abs() <= 1e-4);
        }

        void testLayerDenseBackward(void) {
            TS_TRACE("Starting layer dense backward test");
            
            Layer_Dense dense1(2, 2, 3.0);
            TensorInline weights({2, 2});
            weights.tensor = { 0.07863279, -0.04664191, -0.09444463, -0.04100497};
            dense1.setWeights(weights);


            TensorInline inputs({6, 2});
            inputs.tensor = {-1, -0.5, 0, 0.5, 1, 2, -1, -0.5, 0, 0.5, 1, 2};

            TensorInline output_dweights({2, 2});
            output_dweights.tensor = {7, 2, 2, 6};

            TensorInline output_dinputs({6, 2});
            output_dinputs.tensor = {-0.05531184,  0.11494711, -0.02332096, -0.02050249, -0.01465103, -0.17645457, -0.05531184,  0.11494711, -0.02332096, -0.02050249, -0.01465103, -0.17645457 };

            dense1.forward(inputs);
            dense1.backward(inputs);

            TS_ASSERT((dense1.getDweights() - output_dweights).abs() <= 1e-4);
            TS_ASSERT((dense1.getDinputs() - output_dinputs).abs() <= 1e-4);
        }
};