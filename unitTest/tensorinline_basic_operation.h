#include <cxxtest/TestSuite.h>
#include "../header/tensor_inline.hpp"


class TensorInlineBasicOperation : public CxxTest::TestSuite {
    public:
        void testSetupComparison(void) {
            TS_TRACE("Starting basic comparison tcheck");
            TensorInline a(2, 2, 0);
            TensorInline b(2, 2, 0);
            TensorInline c(2, 2, 2);
            TensorInline d(3, 2, 2);

            TS_ASSERT(a == b);  // Same vector.
            TS_ASSERT(c != d);  // Same value, different size.
            TS_ASSERT(a != c);  // Same size, different value.
        }

        void testAddition(void) {
            TS_TRACE("Starting addition test");
            TensorInline a(2, 2, 1);
            TensorInline b(2, 2, 1);
            TensorInline c(1, 2, 1);
            TensorInline d = a;

            TS_ASSERT_EQUALS(a+b, b+a);    // Tcheck commutative addition.
            TS_ASSERT(c+b == c);           // If not the same size, no addition. 
            TS_ASSERT(b != b+c);           // vectors have same width and second vector is height 1. 

            a += b;
            TS_ASSERT(a == d + b);
        }

        void testSubstration(void) {
            TS_TRACE("Starting substraction tcheck");

            TensorInline a(2, 2, 1);
            TensorInline b(2, 2, 1);
            TensorInline c(1, 2, 1);
            TensorInline d = a;

            TS_ASSERT_EQUALS(a-b, b-a);    // Tcheck commutative substraction.
            TS_ASSERT(c-b == c);           // If not the same size, no substraction. 
            TS_ASSERT(b != b-c);           // vectors have same width and second vector is height 1. 

            a -= b;
            TS_ASSERT(a == d - b);
        }

        void testMultiplication(void) {
            
        }
};


// Download cxxtestgen => sudo apt install cxxtest
// cxxtestgen --error-printer -o unitTest.cpp ../unitTest/tensorinline_addition.h
// g++ -o unitTest unitTest.cpp
// ../unitTest

// faire des actions sur github