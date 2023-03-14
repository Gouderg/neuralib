/* Generated file, do not edit */

#ifndef CXXTEST_RUNNING
#define CXXTEST_RUNNING
#endif

#define _CXXTEST_HAVE_STD
#include <cxxtest/TestListener.h>
#include <cxxtest/TestTracker.h>
#include <cxxtest/TestRunner.h>
#include <cxxtest/RealDescriptions.h>
#include <cxxtest/TestMain.h>
#include <cxxtest/ErrorPrinter.h>

int main( int argc, char *argv[] ) {
 int status;
    CxxTest::ErrorPrinter tmp;
    CxxTest::RealWorldDescription::_worldName = "cxxtest";
    status = CxxTest::Main< CxxTest::ErrorPrinter >( tmp, argc, argv );
    return status;
}
bool suite_TestActivationLinear_init = false;
#include "tests/test_activation_linear.h"

static TestActivationLinear suite_TestActivationLinear;

static CxxTest::List Tests_TestActivationLinear = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_TestActivationLinear( "tests/test_activation_linear.h", 5, "TestActivationLinear", suite_TestActivationLinear, Tests_TestActivationLinear );

static class TestDescription_suite_TestActivationLinear_testActivationLinearForward : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestActivationLinear_testActivationLinearForward() : CxxTest::RealTestDescription( Tests_TestActivationLinear, suiteDescription_TestActivationLinear, 9, "testActivationLinearForward" ) {}
 void runTest() { suite_TestActivationLinear.testActivationLinearForward(); }
} testDescription_suite_TestActivationLinear_testActivationLinearForward;

static class TestDescription_suite_TestActivationLinear_testActivationLinearBackward : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestActivationLinear_testActivationLinearBackward() : CxxTest::RealTestDescription( Tests_TestActivationLinear, suiteDescription_TestActivationLinear, 20, "testActivationLinearBackward" ) {}
 void runTest() { suite_TestActivationLinear.testActivationLinearBackward(); }
} testDescription_suite_TestActivationLinear_testActivationLinearBackward;

#include "tests/test_activation_relu.h"

static TestActivationRelu suite_TestActivationRelu;

static CxxTest::List Tests_TestActivationRelu = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_TestActivationRelu( "tests/test_activation_relu.h", 5, "TestActivationRelu", suite_TestActivationRelu, Tests_TestActivationRelu );

static class TestDescription_suite_TestActivationRelu_testActivationReluForward : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestActivationRelu_testActivationReluForward() : CxxTest::RealTestDescription( Tests_TestActivationRelu, suiteDescription_TestActivationRelu, 9, "testActivationReluForward" ) {}
 void runTest() { suite_TestActivationRelu.testActivationReluForward(); }
} testDescription_suite_TestActivationRelu_testActivationReluForward;

static class TestDescription_suite_TestActivationRelu_testActivationReluBackward : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestActivationRelu_testActivationReluBackward() : CxxTest::RealTestDescription( Tests_TestActivationRelu, suiteDescription_TestActivationRelu, 23, "testActivationReluBackward" ) {}
 void runTest() { suite_TestActivationRelu.testActivationReluBackward(); }
} testDescription_suite_TestActivationRelu_testActivationReluBackward;

#include "tests/test_activation_sigmoid.h"

static TestActivationSigmoid suite_TestActivationSigmoid;

static CxxTest::List Tests_TestActivationSigmoid = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_TestActivationSigmoid( "tests/test_activation_sigmoid.h", 5, "TestActivationSigmoid", suite_TestActivationSigmoid, Tests_TestActivationSigmoid );

static class TestDescription_suite_TestActivationSigmoid_testActivationSigmoidForward : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestActivationSigmoid_testActivationSigmoidForward() : CxxTest::RealTestDescription( Tests_TestActivationSigmoid, suiteDescription_TestActivationSigmoid, 9, "testActivationSigmoidForward" ) {}
 void runTest() { suite_TestActivationSigmoid.testActivationSigmoidForward(); }
} testDescription_suite_TestActivationSigmoid_testActivationSigmoidForward;

static class TestDescription_suite_TestActivationSigmoid_testActivationSigmoidBackward : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestActivationSigmoid_testActivationSigmoidBackward() : CxxTest::RealTestDescription( Tests_TestActivationSigmoid, suiteDescription_TestActivationSigmoid, 24, "testActivationSigmoidBackward" ) {}
 void runTest() { suite_TestActivationSigmoid.testActivationSigmoidBackward(); }
} testDescription_suite_TestActivationSigmoid_testActivationSigmoidBackward;

#include "tests/test_activation_softmax..h"

static TestActivationSoftmax suite_TestActivationSoftmax;

static CxxTest::List Tests_TestActivationSoftmax = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_TestActivationSoftmax( "tests/test_activation_softmax..h", 5, "TestActivationSoftmax", suite_TestActivationSoftmax, Tests_TestActivationSoftmax );

static class TestDescription_suite_TestActivationSoftmax_testActivationSoftmaxForward : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestActivationSoftmax_testActivationSoftmaxForward() : CxxTest::RealTestDescription( Tests_TestActivationSoftmax, suiteDescription_TestActivationSoftmax, 9, "testActivationSoftmaxForward" ) {}
 void runTest() { suite_TestActivationSoftmax.testActivationSoftmaxForward(); }
} testDescription_suite_TestActivationSoftmax_testActivationSoftmaxForward;

static class TestDescription_suite_TestActivationSoftmax_testActivationSoftmaxBackward : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestActivationSoftmax_testActivationSoftmaxBackward() : CxxTest::RealTestDescription( Tests_TestActivationSoftmax, suiteDescription_TestActivationSoftmax, 23, "testActivationSoftmaxBackward" ) {}
 void runTest() { suite_TestActivationSoftmax.testActivationSoftmaxBackward(); }
} testDescription_suite_TestActivationSoftmax_testActivationSoftmaxBackward;

#include "tests/test_loss_absolute_error.h"

static TestLossMeanAbsoluteError suite_TestLossMeanAbsoluteError;

static CxxTest::List Tests_TestLossMeanAbsoluteError = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_TestLossMeanAbsoluteError( "tests/test_loss_absolute_error.h", 5, "TestLossMeanAbsoluteError", suite_TestLossMeanAbsoluteError, Tests_TestLossMeanAbsoluteError );

static class TestDescription_suite_TestLossMeanAbsoluteError_testLossMeanAbsoluteErrorForward : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestLossMeanAbsoluteError_testLossMeanAbsoluteErrorForward() : CxxTest::RealTestDescription( Tests_TestLossMeanAbsoluteError, suiteDescription_TestLossMeanAbsoluteError, 8, "testLossMeanAbsoluteErrorForward" ) {}
 void runTest() { suite_TestLossMeanAbsoluteError.testLossMeanAbsoluteErrorForward(); }
} testDescription_suite_TestLossMeanAbsoluteError_testLossMeanAbsoluteErrorForward;

static class TestDescription_suite_TestLossMeanAbsoluteError_testLossMeanAbsoluteErrorBackward : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestLossMeanAbsoluteError_testLossMeanAbsoluteErrorBackward() : CxxTest::RealTestDescription( Tests_TestLossMeanAbsoluteError, suiteDescription_TestLossMeanAbsoluteError, 44, "testLossMeanAbsoluteErrorBackward" ) {}
 void runTest() { suite_TestLossMeanAbsoluteError.testLossMeanAbsoluteErrorBackward(); }
} testDescription_suite_TestLossMeanAbsoluteError_testLossMeanAbsoluteErrorBackward;

static class TestDescription_suite_TestLossMeanAbsoluteError_testLossMeanAbsoluteErrorAccuracy : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestLossMeanAbsoluteError_testLossMeanAbsoluteErrorAccuracy() : CxxTest::RealTestDescription( Tests_TestLossMeanAbsoluteError, suiteDescription_TestLossMeanAbsoluteError, 63, "testLossMeanAbsoluteErrorAccuracy" ) {}
 void runTest() { suite_TestLossMeanAbsoluteError.testLossMeanAbsoluteErrorAccuracy(); }
} testDescription_suite_TestLossMeanAbsoluteError_testLossMeanAbsoluteErrorAccuracy;

#include "tests/test_loss_binary_crossentropy.h"

static TestLossBinaryCrossEntropy suite_TestLossBinaryCrossEntropy;

static CxxTest::List Tests_TestLossBinaryCrossEntropy = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_TestLossBinaryCrossEntropy( "tests/test_loss_binary_crossentropy.h", 5, "TestLossBinaryCrossEntropy", suite_TestLossBinaryCrossEntropy, Tests_TestLossBinaryCrossEntropy );

static class TestDescription_suite_TestLossBinaryCrossEntropy_testLossBinaryCrossEntropyForward : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestLossBinaryCrossEntropy_testLossBinaryCrossEntropyForward() : CxxTest::RealTestDescription( Tests_TestLossBinaryCrossEntropy, suiteDescription_TestLossBinaryCrossEntropy, 8, "testLossBinaryCrossEntropyForward" ) {}
 void runTest() { suite_TestLossBinaryCrossEntropy.testLossBinaryCrossEntropyForward(); }
} testDescription_suite_TestLossBinaryCrossEntropy_testLossBinaryCrossEntropyForward;

static class TestDescription_suite_TestLossBinaryCrossEntropy_testLossBinaryCrossEntropyBackward : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestLossBinaryCrossEntropy_testLossBinaryCrossEntropyBackward() : CxxTest::RealTestDescription( Tests_TestLossBinaryCrossEntropy, suiteDescription_TestLossBinaryCrossEntropy, 28, "testLossBinaryCrossEntropyBackward" ) {}
 void runTest() { suite_TestLossBinaryCrossEntropy.testLossBinaryCrossEntropyBackward(); }
} testDescription_suite_TestLossBinaryCrossEntropy_testLossBinaryCrossEntropyBackward;

static class TestDescription_suite_TestLossBinaryCrossEntropy_testLossBinaryCrossEntropyAccuracy : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestLossBinaryCrossEntropy_testLossBinaryCrossEntropyAccuracy() : CxxTest::RealTestDescription( Tests_TestLossBinaryCrossEntropy, suiteDescription_TestLossBinaryCrossEntropy, 49, "testLossBinaryCrossEntropyAccuracy" ) {}
 void runTest() { suite_TestLossBinaryCrossEntropy.testLossBinaryCrossEntropyAccuracy(); }
} testDescription_suite_TestLossBinaryCrossEntropy_testLossBinaryCrossEntropyAccuracy;

#include "tests/test_loss_mean_squared_error.h"

static TestLossMeanSquaredError suite_TestLossMeanSquaredError;

static CxxTest::List Tests_TestLossMeanSquaredError = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_TestLossMeanSquaredError( "tests/test_loss_mean_squared_error.h", 5, "TestLossMeanSquaredError", suite_TestLossMeanSquaredError, Tests_TestLossMeanSquaredError );

static class TestDescription_suite_TestLossMeanSquaredError_testLossMeanSquaredErrorForward : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestLossMeanSquaredError_testLossMeanSquaredErrorForward() : CxxTest::RealTestDescription( Tests_TestLossMeanSquaredError, suiteDescription_TestLossMeanSquaredError, 8, "testLossMeanSquaredErrorForward" ) {}
 void runTest() { suite_TestLossMeanSquaredError.testLossMeanSquaredErrorForward(); }
} testDescription_suite_TestLossMeanSquaredError_testLossMeanSquaredErrorForward;

static class TestDescription_suite_TestLossMeanSquaredError_testLossMeanSquaredErrorBackward : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestLossMeanSquaredError_testLossMeanSquaredErrorBackward() : CxxTest::RealTestDescription( Tests_TestLossMeanSquaredError, suiteDescription_TestLossMeanSquaredError, 44, "testLossMeanSquaredErrorBackward" ) {}
 void runTest() { suite_TestLossMeanSquaredError.testLossMeanSquaredErrorBackward(); }
} testDescription_suite_TestLossMeanSquaredError_testLossMeanSquaredErrorBackward;

static class TestDescription_suite_TestLossMeanSquaredError_testLossMeanSquaredErrorAccuracy : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TestLossMeanSquaredError_testLossMeanSquaredErrorAccuracy() : CxxTest::RealTestDescription( Tests_TestLossMeanSquaredError, suiteDescription_TestLossMeanSquaredError, 63, "testLossMeanSquaredErrorAccuracy" ) {}
 void runTest() { suite_TestLossMeanSquaredError.testLossMeanSquaredErrorAccuracy(); }
} testDescription_suite_TestLossMeanSquaredError_testLossMeanSquaredErrorAccuracy;

#include "tests/test_tensorinline_basic_operation.h"

static TensorInlineBasicOperation suite_TensorInlineBasicOperation;

static CxxTest::List Tests_TensorInlineBasicOperation = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_TensorInlineBasicOperation( "tests/test_tensorinline_basic_operation.h", 5, "TensorInlineBasicOperation", suite_TensorInlineBasicOperation, Tests_TensorInlineBasicOperation );

static class TestDescription_suite_TensorInlineBasicOperation_testSetupComparison : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineBasicOperation_testSetupComparison() : CxxTest::RealTestDescription( Tests_TensorInlineBasicOperation, suiteDescription_TensorInlineBasicOperation, 7, "testSetupComparison" ) {}
 void runTest() { suite_TensorInlineBasicOperation.testSetupComparison(); }
} testDescription_suite_TensorInlineBasicOperation_testSetupComparison;

static class TestDescription_suite_TensorInlineBasicOperation_testAddition : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineBasicOperation_testAddition() : CxxTest::RealTestDescription( Tests_TensorInlineBasicOperation, suiteDescription_TensorInlineBasicOperation, 22, "testAddition" ) {}
 void runTest() { suite_TensorInlineBasicOperation.testAddition(); }
} testDescription_suite_TensorInlineBasicOperation_testAddition;

static class TestDescription_suite_TensorInlineBasicOperation_testSubstration : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineBasicOperation_testSubstration() : CxxTest::RealTestDescription( Tests_TensorInlineBasicOperation, suiteDescription_TensorInlineBasicOperation, 52, "testSubstration" ) {}
 void runTest() { suite_TensorInlineBasicOperation.testSubstration(); }
} testDescription_suite_TensorInlineBasicOperation_testSubstration;

static class TestDescription_suite_TensorInlineBasicOperation_testMultiplication : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineBasicOperation_testMultiplication() : CxxTest::RealTestDescription( Tests_TensorInlineBasicOperation, suiteDescription_TensorInlineBasicOperation, 83, "testMultiplication" ) {}
 void runTest() { suite_TensorInlineBasicOperation.testMultiplication(); }
} testDescription_suite_TensorInlineBasicOperation_testMultiplication;

static class TestDescription_suite_TensorInlineBasicOperation_testDivision : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineBasicOperation_testDivision() : CxxTest::RealTestDescription( Tests_TensorInlineBasicOperation, suiteDescription_TensorInlineBasicOperation, 113, "testDivision" ) {}
 void runTest() { suite_TensorInlineBasicOperation.testDivision(); }
} testDescription_suite_TensorInlineBasicOperation_testDivision;

static class TestDescription_suite_TensorInlineBasicOperation_testReshape : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineBasicOperation_testReshape() : CxxTest::RealTestDescription( Tests_TensorInlineBasicOperation, suiteDescription_TensorInlineBasicOperation, 143, "testReshape" ) {}
 void runTest() { suite_TensorInlineBasicOperation.testReshape(); }
} testDescription_suite_TensorInlineBasicOperation_testReshape;

static class TestDescription_suite_TensorInlineBasicOperation_testShape : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineBasicOperation_testShape() : CxxTest::RealTestDescription( Tests_TensorInlineBasicOperation, suiteDescription_TensorInlineBasicOperation, 158, "testShape" ) {}
 void runTest() { suite_TensorInlineBasicOperation.testShape(); }
} testDescription_suite_TensorInlineBasicOperation_testShape;

static class TestDescription_suite_TensorInlineBasicOperation_testRecopy : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineBasicOperation_testRecopy() : CxxTest::RealTestDescription( Tests_TensorInlineBasicOperation, suiteDescription_TensorInlineBasicOperation, 171, "testRecopy" ) {}
 void runTest() { suite_TensorInlineBasicOperation.testRecopy(); }
} testDescription_suite_TensorInlineBasicOperation_testRecopy;

#include "tests/test_tensorinline_complex_operation.h"

static TensorInlineComplexeOperation suite_TensorInlineComplexeOperation;

static CxxTest::List Tests_TensorInlineComplexeOperation = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_TensorInlineComplexeOperation( "tests/test_tensorinline_complex_operation.h", 4, "TensorInlineComplexeOperation", suite_TensorInlineComplexeOperation, Tests_TensorInlineComplexeOperation );

static class TestDescription_suite_TensorInlineComplexeOperation_testTransposition : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineComplexeOperation_testTransposition() : CxxTest::RealTestDescription( Tests_TensorInlineComplexeOperation, suiteDescription_TensorInlineComplexeOperation, 7, "testTransposition" ) {}
 void runTest() { suite_TensorInlineComplexeOperation.testTransposition(); }
} testDescription_suite_TensorInlineComplexeOperation_testTransposition;

static class TestDescription_suite_TensorInlineComplexeOperation_testSum : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineComplexeOperation_testSum() : CxxTest::RealTestDescription( Tests_TensorInlineComplexeOperation, suiteDescription_TensorInlineComplexeOperation, 22, "testSum" ) {}
 void runTest() { suite_TensorInlineComplexeOperation.testSum(); }
} testDescription_suite_TensorInlineComplexeOperation_testSum;

static class TestDescription_suite_TensorInlineComplexeOperation_testSqrt : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineComplexeOperation_testSqrt() : CxxTest::RealTestDescription( Tests_TensorInlineComplexeOperation, suiteDescription_TensorInlineComplexeOperation, 41, "testSqrt" ) {}
 void runTest() { suite_TensorInlineComplexeOperation.testSqrt(); }
} testDescription_suite_TensorInlineComplexeOperation_testSqrt;

static class TestDescription_suite_TensorInlineComplexeOperation_testAbs : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineComplexeOperation_testAbs() : CxxTest::RealTestDescription( Tests_TensorInlineComplexeOperation, suiteDescription_TensorInlineComplexeOperation, 55, "testAbs" ) {}
 void runTest() { suite_TensorInlineComplexeOperation.testAbs(); }
} testDescription_suite_TensorInlineComplexeOperation_testAbs;

static class TestDescription_suite_TensorInlineComplexeOperation_testDotProduct : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineComplexeOperation_testDotProduct() : CxxTest::RealTestDescription( Tests_TensorInlineComplexeOperation, suiteDescription_TensorInlineComplexeOperation, 67, "testDotProduct" ) {}
 void runTest() { suite_TensorInlineComplexeOperation.testDotProduct(); }
} testDescription_suite_TensorInlineComplexeOperation_testDotProduct;

static class TestDescription_suite_TensorInlineComplexeOperation_testExponential : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineComplexeOperation_testExponential() : CxxTest::RealTestDescription( Tests_TensorInlineComplexeOperation, suiteDescription_TensorInlineComplexeOperation, 80, "testExponential" ) {}
 void runTest() { suite_TensorInlineComplexeOperation.testExponential(); }
} testDescription_suite_TensorInlineComplexeOperation_testExponential;

static class TestDescription_suite_TensorInlineComplexeOperation_testClipped : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineComplexeOperation_testClipped() : CxxTest::RealTestDescription( Tests_TensorInlineComplexeOperation, suiteDescription_TensorInlineComplexeOperation, 89, "testClipped" ) {}
 void runTest() { suite_TensorInlineComplexeOperation.testClipped(); }
} testDescription_suite_TensorInlineComplexeOperation_testClipped;

static class TestDescription_suite_TensorInlineComplexeOperation_testSign : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineComplexeOperation_testSign() : CxxTest::RealTestDescription( Tests_TensorInlineComplexeOperation, suiteDescription_TensorInlineComplexeOperation, 101, "testSign" ) {}
 void runTest() { suite_TensorInlineComplexeOperation.testSign(); }
} testDescription_suite_TensorInlineComplexeOperation_testSign;

static class TestDescription_suite_TensorInlineComplexeOperation_testStandardDeviation : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineComplexeOperation_testStandardDeviation() : CxxTest::RealTestDescription( Tests_TensorInlineComplexeOperation, suiteDescription_TensorInlineComplexeOperation, 112, "testStandardDeviation" ) {}
 void runTest() { suite_TensorInlineComplexeOperation.testStandardDeviation(); }
} testDescription_suite_TensorInlineComplexeOperation_testStandardDeviation;

#include <cxxtest/Root.cpp>
const char* CxxTest::RealWorldDescription::_worldName = "cxxtest";
