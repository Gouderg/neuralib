#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"

class TensorInlineComplexeOperation: public CxxTest::TestSuite {

    public:
        void testTransposition(void) {
            TS_TRACE("Starting transposition test");

            TensorInline v1({2, 3, false, 1});
            TensorInline v2({3, 2, false, 1});

            TensorInline v3({2, 2, false, 1});
            TensorInline v4({2, 2, false, 1});

            TS_ASSERT_EQUALS(v1, v2.transposate());
            TS_ASSERT_EQUALS(v2, v1.transposate());
            TS_ASSERT_EQUALS(v3, v4.transposate());
            TS_ASSERT(v2 != v4.transposate());
        }

        void testSum(void) {
            TS_TRACE("Starting sum test");

            TensorInline v1({2, 2, false, 0.5});
            TensorInline v2({3, 2, false, 4});
            TensorInline v3({3, 2, false, -4.0});
            TensorInline v4({2, 2, false, -0.5});

            std::vector<double> v1bis = {0.5, 0.5, 0.5, 0.5};
            std::vector<double> v4bis = {-0.5, -0.5, -0.5, -0.5};


            double res1 = 2;
            double res2 = 24;
            double res3 = -24;
            double res4 = -2;

            TS_ASSERT_EQUALS(TensorInline::sum(v1), res1);
            TS_ASSERT_EQUALS(TensorInline::sum(v2), res2);
            TS_ASSERT_EQUALS(TensorInline::sum(v3), res3);
            TS_ASSERT_EQUALS(TensorInline::sum(v4), res4);
            TS_ASSERT_EQUALS(TensorInline::sum(v1bis), res1);
            TS_ASSERT_EQUALS(TensorInline::sum(v4bis), res4);

        }

        void testSqrt(void) {
            TS_TRACE("Starting sqrt test");

            TensorInline v1({2, 3, false, 4.0});
            TensorInline v2({2, 3, false, 2.0});
            TensorInline v3({2, 3});
            TensorInline v4({2, 3});
            v3.tensor = {-2, -1, 0, 4, 100, 0};
            v4.tensor = {-2, -1, 0, 2, 10, 0};

            TS_ASSERT_EQUALS(v1.sqrt(), v2);
            TS_ASSERT_EQUALS(v3.sqrt(), v4);
        }

        void testAbs(void) {
            TS_TRACE("Starting abs test");

            TensorInline v1({2, 3, false, -4.0});
            TensorInline v2({2, 3, false, 4.0});
            TensorInline v3({2, 3, false, -0.5});
            TensorInline v4({2, 3, false, 0.5});

            TS_ASSERT_EQUALS(v1.abs(), v2);
            TS_ASSERT_EQUALS(v3.abs(), v4);
        }
        
        void testDotProduct(void) {
            TS_TRACE("Starting dot product test");

            TensorInline v1({2, 3, false, -4.0});
            TensorInline v2({3, 2, false, 4.0});
            TensorInline v3({2, 2, false, -48});
            std::vector<double> v4 = {4, 4, 4};
            TensorInline v5({2, 3, false, -48});

            TS_ASSERT_EQUALS(TensorInline::dot(v1, v2), v3); 
            TS_ASSERT_EQUALS(TensorInline::dot(v1, v4), v5); 
        }

        void testExponential(void) {
            TS_TRACE("Starting exponential test");

            TensorInline v1({2, 3, false, 0.0});
            TensorInline res({2, 3, false, 1.0});

            TS_ASSERT(TensorInline::exp(v1) == res);
        }

        void testClipped(void) {
            TS_TRACE("Starting clipped test");

            TensorInline v1({2, 2, false, 3.0});
            TensorInline v2({2, 2, false, 1.0});
            TensorInline res({2, 2, false, 2.0});
            double range = 2.0;

            TS_ASSERT_EQUALS(TensorInline::clip(v1, range, 10.0 - range), v1);    // Clipped.
            TS_ASSERT_EQUALS(TensorInline::clip(v2, range, 10.0 - range), res);   // Not clipped.
        }    

        void testSign(void) {
            TS_TRACE("Starting sign test");
            double res1 = 3.0 / 1000.0;
            double res2 = 0e-100 / 1000.0;
            double res3 = -4.0 / 1000.0;

            TS_ASSERT_EQUALS(TensorInline::sign(res1), 1);
            TS_ASSERT_EQUALS(TensorInline::sign(res2), 0);
            TS_ASSERT_EQUALS(TensorInline::sign(res3), -1);
        }

        void testStandardDeviation(void) {
            TS_TRACE("Starting standard deviation test");

            TensorInline v1({4, 1, false, 1.0});

            TensorInline v2({16, 1, false, 0.0});
            v2.tensor = {1, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

            TensorInline v3({4, 1, false, 1.0});
            v3.tensor = {1, 1, 3, 3};

            TS_ASSERT_EQUALS(TensorInline::standard_deviation(v1), 0);
            TS_ASSERT_EQUALS(TensorInline::standard_deviation(v2), 0.5);
            TS_ASSERT_EQUALS(TensorInline::standard_deviation(v3), 1.0);
        }

        void testMean(void) {
            TS_TRACE("Starting mean test");

            TensorInline v1({4, 1, false, 1.0});
            double res1 = 1.0;

            TensorInline v2({16, 1, false, 0.0});
            v2.tensor = {1, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
            double res2 = 2.0;

            TensorInline v3({4, 1, false, 0.0});
            v3.tensor = {1, 1, -2, 3};
            double res3 = 0.75;

            TS_ASSERT_EQUALS(TensorInline::mean(v1),res1);
            TS_ASSERT_EQUALS(TensorInline::mean(v2),res2);
            TS_ASSERT_EQUALS(TensorInline::mean(v3),res3);
        }

        void testRound(void) {
            TS_TRACE("Starting round test");

            TS_ASSERT_EQUALS(TensorInline::round(0.1), 0);
            TS_ASSERT_EQUALS(TensorInline::round(0.7), 1);
            TS_ASSERT_EQUALS(TensorInline::round(1.1), 1);
            TS_ASSERT_EQUALS(TensorInline::round(2.1), 2);
            TS_ASSERT_EQUALS(TensorInline::round(-2.1), -2);
            TS_ASSERT_EQUALS(TensorInline::round(-2.7), -3);
        }
};