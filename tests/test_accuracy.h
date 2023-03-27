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
};