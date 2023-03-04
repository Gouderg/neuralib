#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"


class TensorInlineComplexeOperation: public CxxTest::TestSuite {

    public:
        void testTransposition(void) {
            TensorInline v1({2, 3, false, 1});
            TensorInline v2({3, 2, false, 1});

            TensorInline v3({2, 2, false, 1});
            TensorInline v4({2, 2, false, 1});

            TS_ASSERT_EQUALS(v1, v2.transposate());
            TS_ASSERT_EQUALS(v3, v4.transposate());
            TS_ASSERT(v2 != v4.transposate());
        }

        void testSum(void) {
            TensorInline v1({2, 2, false, 0.5});
            TensorInline v2({3, 2, false, 4});
            double res1 = 2;
            double res2 = 24;

            TS_ASSERT_EQUALS(TensorInline::sum(v1), res1);
            TS_ASSERT_EQUALS(TensorInline::sum(v2), res2);
        }
        
           
};