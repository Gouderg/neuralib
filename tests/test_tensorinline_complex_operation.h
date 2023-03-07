#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"

#define CXXTEST_HAVE_EH

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

            double res1 = 2;
            double res2 = 24;
            double res3 = -24;
            double res4 = -2;

            TS_ASSERT_EQUALS(TensorInline::sum(v1), res1);
            TS_ASSERT_EQUALS(TensorInline::sum(v2), res2);
            TS_ASSERT_EQUALS(TensorInline::sum(v3), res3);
            TS_ASSERT_EQUALS(TensorInline::sum(v4), res4);
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
};