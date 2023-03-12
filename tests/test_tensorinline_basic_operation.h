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
            
            TS_ASSERT(a <= 3.0);    // Should be true.
            TS_ASSERT(!(c <= 0.0)); // Should be false.
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

            // res = n + v1 
            TS_ASSERT_EQUALS(n + v1, res);

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

            // res3 = n - v1
            TensorInline res3({2, 2, false, 2});
            TS_ASSERT_EQUALS(n - v1, res3);

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

            // res = n * v1
            TS_ASSERT_EQUALS(n * v1, res);

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

            // res3 = n / v1
            TensorInline res3({2, 2, false, 2});
            TS_ASSERT_EQUALS(n / v1, res3);

            // v1 /= v2
            v1 /= v2;
            TensorInline res1({2, 2, false, 1});
            TS_ASSERT(v1 == res1);

            // v1 /= n
            v1 /= n;
            TensorInline res2({2, 2, false, 0.25});
            TS_ASSERT_EQUALS(v1, res2);
        }

        void testReshape(void) {
            TS_TRACE("Starting reshape test");

            TensorInline v1({4, 5});

            v1.reshape(2, 10);
            TS_ASSERT(v1.getHeight() == 2 && v1.getWidth() == 10);
            
            v1.reshape(-1, 1);
            TS_ASSERT(v1.getHeight() == 20 && v1.getWidth() == 1);

            v1.reshape(1, -1);
            TS_ASSERT(v1.getHeight() == 1 && v1.getWidth() == 20);
        }

        void testRecopy(void) {
            TS_TRACE("Starting contructor copy test");

            TensorInline v1({4, 5});

            TensorInline v2 = v1;
            v1.reshape(2, 10);

            TS_ASSERT(v1.getHeight() != v2.getHeight());

            v2.reshape(2, 10);
            TS_ASSERT(v1.getHeight() == v2.getHeight());

        }
};