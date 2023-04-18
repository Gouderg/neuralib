#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"
#include "../header/accuracy.hpp"

class TestAccuracy: public CxxTest::TestSuite {

    public:

        void testAccuracyCategorical(void) {
            TS_TRACE("Start accuracy categorical trace");
            TensorInline inputs ({2, 3});
            inputs.tensor = {1, 3, 2, 4, 7, 6};

            TensorInline y({2, 3});
            y.tensor = {1, 2, 3, 4, 5, 6};

            double res = 0.5;

            Accuracy_Categorical acc(false);
            acc.init(y);

            TS_ASSERT_EQUALS(acc.calculate(inputs, y), res);
        }

        void testAccuracyRegression(void) {
            TS_TRACE("Start accuracy regression trace");

            TensorInline inputs ({2, 3});
            inputs.tensor = {1, 3, 2, 4, 7, 6};

            TensorInline y({2, 3});
            y.tensor = {1, 2, 3, 4, 5, 6};

            double res = 0.5;

            Accuracy_Regression acc(2);
            acc.init(y, true);

            TS_ASSERT_EQUALS(acc.calculate(inputs, y), res);
        }

        void testAccuracyBinary(void) {
            TS_TRACE("Starting Accuracy binary test");

            TensorInline y_test({12, 1});
            y_test.tensor = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1};

            TensorInline y_true({12, 1});
            y_true.tensor = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

            Accuracy_Binary accuracy_function;

            double accuracy = accuracy_function.calculate(y_test, y_true);
            TS_ASSERT_EQUALS(accuracy, 1.0);

            y_test.tensor = {0.1, 0.8, 0.2, 0.3, 0.9, 0.2, 0.2, 0.1, 0.2, 0.9, 2.0, 0.0};
            accuracy = accuracy_function.calculate(y_test, y_true);
            TS_ASSERT_EQUALS(accuracy, 0.5);

        }
};