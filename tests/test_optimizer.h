#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/optimizer.hpp"

class TestOptimizer: public CxxTest::TestSuite {

    public:
        void testOptimizerSGD(void) {
            TS_TRACE("Starting optimizer_sgd test");

            TensorInline inputs ({6, 2});
            inputs.tensor = {0.1, 0.3, 0.6, 0.3, 0.5, 0.2, 0.333333, 0.333333, 0.333333, 0.1, 0.8, 0.1};

            Optimizer_SGD optimizer_without_momentum = Optimizer_SGD(1.5, 0.2, 0.0);
            Optimizer_SGD optimizer_with_momentum = Optimizer_SGD(1.5, 0.2, 1.5);

            Layer_Dense dense_without_momentum({2, 2});
            TensorInline weights({2, 2});
            weights.tensor = {0.17640524, 0.04001572, 0.0978738, 0.22408931};
            dense_without_momentum.setWeights(weights);

            Layer_Dense dense_with_momentum({2, 2});
            TensorInline weights2({2, 2});
            weights2.tensor = {0.1867558, -0.09772779, 0.09500884, -0.01513572};
            dense_with_momentum.setWeights(weights2);

            TensorInline expected_weights_dense_without_momentum({2, 2});
            expected_weights_dense_without_momentum.tensor = {-2.0469275, -0.7616506, -0.7037925, -0.30257696};

            TensorInline expected_weights_dense_with_momentum({2, 2});
            expected_weights_dense_with_momentum.tensor = {-7.594908, -2.90356, -2.7108233, -1.8584676};

            TensorInline expected_weights_momentum_dense_with_momentum({2, 2});
            expected_weights_momentum_dense_with_momentum.tensor = {-5.5583315, -2.004166, -2.004166, -1.3166656};

            TensorInline expected_biases_dense_without_momentum({1, 2});
            expected_biases_dense_without_momentum.tensor = {-3.999999, -1.9999995};

            TensorInline expected_biases_dense_with_momentum({1, 2});
            expected_biases_dense_with_momentum.tensor = {-13.999996, -6.999998};

            TensorInline expected_biases_momentum_dense_with_momentum({1, 2});
            expected_biases_momentum_dense_with_momentum.tensor = {-9.9999975, -4.99999875};

            dense_without_momentum.forward(inputs);
            dense_without_momentum.backward(inputs);
            dense_with_momentum.forward(inputs);
            dense_with_momentum.backward(inputs);

            optimizer_without_momentum.pre_update_params();
            optimizer_without_momentum.update_params(dense_without_momentum);

            optimizer_with_momentum.pre_update_params();
            optimizer_with_momentum.update_params(dense_with_momentum);
            optimizer_with_momentum.update_params(dense_with_momentum); // Need to apply two times to have the momentum value.

            TS_ASSERT((dense_without_momentum.getWeights() - expected_weights_dense_without_momentum).abs() <= 1e-6);
            TS_ASSERT((dense_with_momentum.getWeights() - expected_weights_dense_with_momentum).abs() <= 1e-6);
            TS_ASSERT((dense_with_momentum.getWeightMomentum() - expected_weights_momentum_dense_with_momentum).abs() <= 1e-6);
            TS_ASSERT((dense_without_momentum.getBiases() - expected_biases_dense_without_momentum).abs() <= 1e5);
            TS_ASSERT((dense_with_momentum.getBiases() - expected_biases_dense_with_momentum).abs() <= 1e-5);
            TS_ASSERT((dense_with_momentum.getBiasMomentum() - expected_biases_momentum_dense_with_momentum).abs() <= 1e-5);
        }

        void testOptimizerAdagrad(void) {
            TS_TRACE("Starting optimizer_adagrad test");

            TensorInline inputs ({6, 2});
            inputs.tensor = {0.1, 0.3, 0.6, 0.3, 0.5, 0.2, 0.333333, 0.333333, 0.333333, 0.1, 0.8, 0.1};

            Optimizer_Adagrad optimizer_without_cache = Optimizer_Adagrad(1.5, 0.2, 0.0);
            Optimizer_Adagrad optimizer_with_cache = Optimizer_Adagrad(1.5, 0.2, 1.5);

            Layer_Dense dense_without_cache({2, 2});
            TensorInline weights({2, 2});
            weights.tensor = {0.17640524, 0.04001572, 0.0978738, 0.22408931};
            dense_without_cache.setWeights(weights);

            Layer_Dense dense_with_cache({2, 2});
            TensorInline weights2({2, 2});
            weights2.tensor = {0.1867558, -0.09772779, 0.09500884, -0.01513572};
            dense_with_cache.setWeights(weights2);

            TensorInline expected_weights_dense_without_cache({2, 2});
            expected_weights_dense_without_cache.tensor = {-1.3235947, -1.459984, -1.4021258, -1.2759103};

            TensorInline expected_weights_dense_with_cache({2, 2});
            expected_weights_dense_with_cache.tensor = {-1.177022, -0.84715176, -0.65441513, -0.56343806};

            TensorInline expected_weights_cache_dense_with_cache({2, 2});
            expected_weights_cache_dense_with_cache.tensor = {4.3939624, 0.5712612, 0.5712612, 0.2465577};

            TensorInline expected_biases_dense_without_cache({1, 2});
            expected_biases_dense_without_cache.tensor = {-1.5, -1.4999999};

            TensorInline expected_biases_dense_with_cache({1, 2});
            expected_biases_dense_with_cache.tensor = {-1.7188351, -1.2966162};

            TensorInline expected_biases_cache_dense_with_cache({1, 2});
            expected_biases_cache_dense_with_cache.tensor = {14.222215, 3.5555537};


            dense_without_cache.forward(inputs);
            dense_without_cache.backward(inputs);
            dense_with_cache.forward(inputs);
            dense_with_cache.backward(inputs);

            optimizer_without_cache.pre_update_params();
            optimizer_without_cache.update_params(dense_without_cache);

            optimizer_with_cache.pre_update_params();
            optimizer_with_cache.update_params(dense_with_cache);
            optimizer_with_cache.update_params(dense_with_cache); // Need to apply two times to have the cache value.

            TS_ASSERT((dense_without_cache.getWeights() - expected_weights_dense_without_cache).abs() <= 1e-6);
            TS_ASSERT((dense_with_cache.getWeights() - expected_weights_dense_with_cache).abs() <= 1e-6);
            TS_ASSERT((dense_with_cache.getWeightCache() - expected_weights_cache_dense_with_cache).abs() <= 1e-6);
            TS_ASSERT((dense_without_cache.getBiases() - expected_biases_dense_without_cache).abs() <= 1e5);
            TS_ASSERT((dense_with_cache.getBiases() - expected_biases_dense_with_cache).abs() <= 1e-5);
            TS_ASSERT((dense_with_cache.getBiasCache() - expected_biases_cache_dense_with_cache).abs() <= 1e-5);
        }

        void testOptimizerRMSProp(void) {
            TS_TRACE("Starting optimizer RMSProp test");

            TensorInline inputs ({6, 2});
            inputs.tensor = {0.1, 0.3, 0.6, 0.3, 0.5, 0.2, 0.333333, 0.333333, 0.333333, 0.1, 0.8, 0.1};

            Optimizer_RMSprop optimizer_with_cache = Optimizer_RMSprop(1.5, 0.2, 1.5, 0.04);

            Layer_Dense dense_with_cache({2, 2});
            TensorInline weights2({2, 2});
            weights2.tensor = {0.17640524, 0.04001572, 0.0978738, 0.22408931};
            dense_with_cache.setWeights(weights2);

            TensorInline expected_weights_dense_with_cache({2, 2});
            expected_weights_dense_with_cache.tensor = {-1.3225117, -0.7502634, -0.69240534, -0.34607565};

            TensorInline expected_weights_cache_dense_with_cache({2, 2});
            expected_weights_cache_dense_with_cache.tensor = {2.1934662, 0.2851736, 0.2851736, 0.12308159};

            TensorInline expected_biases_dense_with_cache({1, 2});
            expected_biases_dense_with_cache.tensor = {-1.9330678, -1.4188063};

            TensorInline expected_biases_cache_dense_with_cache({1, 2});
            expected_biases_cache_dense_with_cache.tensor = {7.09972978, 1.77493245};

            dense_with_cache.forward(inputs);
            dense_with_cache.backward(inputs);

            optimizer_with_cache.pre_update_params();
            optimizer_with_cache.update_params(dense_with_cache);
            optimizer_with_cache.update_params(dense_with_cache); // Need to apply two times to have the cache value.

            TS_ASSERT((dense_with_cache.getWeights() - expected_weights_dense_with_cache).abs() <= 1e-6);
            TS_ASSERT((dense_with_cache.getWeightCache() - expected_weights_cache_dense_with_cache).abs() <= 1e-6);
            TS_ASSERT((dense_with_cache.getBiases() - expected_biases_dense_with_cache).abs() <= 1e-5);
            TS_ASSERT((dense_with_cache.getBiasCache() - expected_biases_cache_dense_with_cache).abs() <= 1e-5);
        }

        void testOptimizerAdam(void) {
            TS_TRACE("Starting optimizer Adam test");

            TensorInline inputs ({6, 2});
            inputs.tensor = {0.1, 0.3, 0.6, 0.3, 0.5, 0.2, 0.333333, 0.333333, 0.333333, 0.1, 0.8, 0.1};

            Optimizer_Adam optimizer_with_cache = Optimizer_Adam(1.5, 0.2, 1.5, 0.09, 0.002);

            Layer_Dense dense({2, 2});
            TensorInline weights2({2, 2});
            weights2.tensor = {0.17640524, 0.04001572, 0.0978738, 0.22408931};
            dense.setWeights(weights2);

            TensorInline expected_weights_dense({2, 2});
            expected_weights_dense.tensor = {-1.3813467, -0.7834294, -0.72557133, -0.37048542};

            TensorInline expected_weights_cache_dense({2, 2});
            expected_weights_cache_dense.tensor = {2.1969726, 0.28562948, 0.28562948, 0.12327836};

            TensorInline expected_weights_momentum_dense({2, 2});
            expected_weights_momentum_dense.tensor = {1.4702157, 0.53011525, 0.53011525, 0.3482669};

            TensorInline expected_biases_dense_with_cache({1, 2});
            expected_biases_dense_with_cache.tensor = {-2.0057309, -1.4749322};

            TensorInline expected_biases_cache_dense_with_cache({1, 2});
            expected_biases_cache_dense_with_cache.tensor = {7.11107911, 1.77776978};

            TensorInline expected_biases_momentum_dense({1, 2});
            expected_biases_momentum_dense.tensor = {2.64506601, 1.322533};

            dense.forward(inputs);
            dense.backward(inputs);

            optimizer_with_cache.pre_update_params();
            optimizer_with_cache.update_params(dense);
            optimizer_with_cache.update_params(dense); // Need to apply two times to have the cache value.

            TS_ASSERT((dense.getWeights() - expected_weights_dense).abs() <= 1e-6);
            TS_ASSERT((dense.getWeightCache() - expected_weights_cache_dense).abs() <= 1e-6);
            TS_ASSERT((dense.getWeightMomentum() - expected_weights_momentum_dense).abs() <= 1e-6);
            TS_ASSERT((dense.getBiases() - expected_biases_dense_with_cache).abs() <= 1e-5);
            TS_ASSERT((dense.getBiasCache() - expected_biases_cache_dense_with_cache).abs() <= 1e-5);
            TS_ASSERT((dense.getBiasMomentum() - expected_biases_momentum_dense).abs() <= 1e-6);

        }

};