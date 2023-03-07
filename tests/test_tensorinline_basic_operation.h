#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"


class TensorInlineBasicOperation : public CxxTest::TestSuite {
    public:
        void testSetupComparison(void) {
            TS_TRACE("Starting basic comparison tcheck");
            TensorInline a({2, 2});
            TensorInline b({2, 2});
            TensorInline c({3, 2, false, 1});
            TensorInline d({2, 3, false, 1});


            TS_ASSERT(a == b);  // Same vector.
            TS_ASSERT(c != d);  // Same value, different size.
        }

        void testAddition(void) {
            TS_TRACE("Starting addition test");
            TensorInline v1({2, 2, false, 2});
            TensorInline v2({2, 2, false, 2});
            TensorInline c({1, 2, false, 2});
            double n = 4.0;

            // res = v1 + v2.
            TS_ASSERT_EQUALS(v1+v2, v2+v1);  // Tcheck commutative addition.
            TS_ASSERT(v2 != v2+c);           // vectors have same width and second vector is height 1. 

            // res = v1 + n
            TensorInline res({2, 2, false, 6});
            TS_ASSERT_EQUALS(v1 + n, res);

            // v1 += v2
            v1 += v2;
            TensorInline res1({2, 2, false, 4});
            TS_ASSERT(v1 == res1);

            // v1 += n
            v1 += n;
            TensorInline res2({2, 2, false, 8});
            TS_ASSERT_EQUALS(v1, res2);

        }

        void testSubstration(void) {
            TS_TRACE("Starting substraction tcheck");

            TensorInline v1({2, 2, false, 2});
            TensorInline v2({2, 2, false, 2});
            TensorInline c({1, 2, false, 2});
            double n = 4.0;

            // res = v1 - v2.
            TS_ASSERT_EQUALS(v1-v2, v2-v1);  // Tcheck commutative substraction.
            TS_ASSERT(v2 != v2-c);           // vectors have same width and second vector is height 1. 

            // res = v1 - n
            TensorInline res({2, 2, false, -2});
            TS_ASSERT_EQUALS(v1 - n, res);

            // v1 -= v2
            v1 -= v2;
            TensorInline res1({2, 2, false, 0});
            TS_ASSERT(v1 == res1);

            // v1 -= n
            v1 -= n;
            TensorInline res2({2, 2, false, -4});
            TS_ASSERT_EQUALS(v1, res2);
        }

        void testMultiplication(void) {
            TS_TRACE("Starting multiplication tcheck");

            TensorInline v1({2, 2, false, 2});
            TensorInline v2({2, 2, false, 2});
            TensorInline c({1, 2, false, 2});
            double n = 4.0;

            // res = v1 * v2.
            TS_ASSERT_EQUALS(v1*v2, v2*v1);  // Tcheck commutative multiplication.
            TS_ASSERT(v2 != v2*c);           // vectors have same width and second vector is height 1. 

            // res = v1 * n
            TensorInline res({2, 2, false, 8});
            TS_ASSERT_EQUALS(v1 * n, res);

            // v1 *= v2
            v1 += v2;
            TensorInline res1({2, 2, false, 4});
            TS_ASSERT(v1 == res1);

            // v1 *= n
            v1 *= n;
            TensorInline res2({2, 2, false, 16});
            TS_ASSERT_EQUALS(v1, res2);
        }

        void testDivision(void) {
            TS_TRACE("Starting division tcheck");

            TensorInline v1({2, 2, false, 2});
            TensorInline v2({2, 2, false, 2});
            TensorInline c({1, 2, false, 2});
            double n = 4.0;

            // res = v1 / v2.
            TS_ASSERT(v2 != v2/c);           // vectors have same width and second vector is height 1. 

            // res = v1 / n
            TensorInline res({2, 2, false, 0.5});
            TS_ASSERT_EQUALS(v1 / n, res);

            // v1 /= v2
            v1 /= v2;
            TensorInline res1({2, 2, false, 1});
            TS_ASSERT(v1 == res1);

            // v1 /= n
            v1 /= n;
            TensorInline res2({2, 2, false, 0.25});
            TS_ASSERT_EQUALS(v1, res2);
        }
};