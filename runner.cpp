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
bool suite_TensorInlineBasicOperation_init = false;
#include "unitTest/tensorinline_basic_operation.h"

static TensorInlineBasicOperation suite_TensorInlineBasicOperation;

static CxxTest::List Tests_TensorInlineBasicOperation = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_TensorInlineBasicOperation( "unitTest/tensorinline_basic_operation.h", 5, "TensorInlineBasicOperation", suite_TensorInlineBasicOperation, Tests_TensorInlineBasicOperation );

static class TestDescription_suite_TensorInlineBasicOperation_testSetupComparison : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineBasicOperation_testSetupComparison() : CxxTest::RealTestDescription( Tests_TensorInlineBasicOperation, suiteDescription_TensorInlineBasicOperation, 7, "testSetupComparison" ) {}
 void runTest() { suite_TensorInlineBasicOperation.testSetupComparison(); }
} testDescription_suite_TensorInlineBasicOperation_testSetupComparison;

static class TestDescription_suite_TensorInlineBasicOperation_testAddition : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineBasicOperation_testAddition() : CxxTest::RealTestDescription( Tests_TensorInlineBasicOperation, suiteDescription_TensorInlineBasicOperation, 19, "testAddition" ) {}
 void runTest() { suite_TensorInlineBasicOperation.testAddition(); }
} testDescription_suite_TensorInlineBasicOperation_testAddition;

static class TestDescription_suite_TensorInlineBasicOperation_testSubstration : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineBasicOperation_testSubstration() : CxxTest::RealTestDescription( Tests_TensorInlineBasicOperation, suiteDescription_TensorInlineBasicOperation, 47, "testSubstration" ) {}
 void runTest() { suite_TensorInlineBasicOperation.testSubstration(); }
} testDescription_suite_TensorInlineBasicOperation_testSubstration;

static class TestDescription_suite_TensorInlineBasicOperation_testMultiplication : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineBasicOperation_testMultiplication() : CxxTest::RealTestDescription( Tests_TensorInlineBasicOperation, suiteDescription_TensorInlineBasicOperation, 75, "testMultiplication" ) {}
 void runTest() { suite_TensorInlineBasicOperation.testMultiplication(); }
} testDescription_suite_TensorInlineBasicOperation_testMultiplication;

static class TestDescription_suite_TensorInlineBasicOperation_testDivision : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineBasicOperation_testDivision() : CxxTest::RealTestDescription( Tests_TensorInlineBasicOperation, suiteDescription_TensorInlineBasicOperation, 103, "testDivision" ) {}
 void runTest() { suite_TensorInlineBasicOperation.testDivision(); }
} testDescription_suite_TensorInlineBasicOperation_testDivision;

#include "unitTest/tensorinline_complex_operation.h"

static TensorInlineComplexeOperation suite_TensorInlineComplexeOperation;

static CxxTest::List Tests_TensorInlineComplexeOperation = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_TensorInlineComplexeOperation( "unitTest/tensorinline_complex_operation.h", 5, "TensorInlineComplexeOperation", suite_TensorInlineComplexeOperation, Tests_TensorInlineComplexeOperation );

static class TestDescription_suite_TensorInlineComplexeOperation_testTransposition : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineComplexeOperation_testTransposition() : CxxTest::RealTestDescription( Tests_TensorInlineComplexeOperation, suiteDescription_TensorInlineComplexeOperation, 8, "testTransposition" ) {}
 void runTest() { suite_TensorInlineComplexeOperation.testTransposition(); }
} testDescription_suite_TensorInlineComplexeOperation_testTransposition;

static class TestDescription_suite_TensorInlineComplexeOperation_testSum : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineComplexeOperation_testSum() : CxxTest::RealTestDescription( Tests_TensorInlineComplexeOperation, suiteDescription_TensorInlineComplexeOperation, 23, "testSum" ) {}
 void runTest() { suite_TensorInlineComplexeOperation.testSum(); }
} testDescription_suite_TensorInlineComplexeOperation_testSum;

static class TestDescription_suite_TensorInlineComplexeOperation_testSqrt : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineComplexeOperation_testSqrt() : CxxTest::RealTestDescription( Tests_TensorInlineComplexeOperation, suiteDescription_TensorInlineComplexeOperation, 42, "testSqrt" ) {}
 void runTest() { suite_TensorInlineComplexeOperation.testSqrt(); }
} testDescription_suite_TensorInlineComplexeOperation_testSqrt;

static class TestDescription_suite_TensorInlineComplexeOperation_testAbs : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineComplexeOperation_testAbs() : CxxTest::RealTestDescription( Tests_TensorInlineComplexeOperation, suiteDescription_TensorInlineComplexeOperation, 56, "testAbs" ) {}
 void runTest() { suite_TensorInlineComplexeOperation.testAbs(); }
} testDescription_suite_TensorInlineComplexeOperation_testAbs;

static class TestDescription_suite_TensorInlineComplexeOperation_testDotProduct : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_TensorInlineComplexeOperation_testDotProduct() : CxxTest::RealTestDescription( Tests_TensorInlineComplexeOperation, suiteDescription_TensorInlineComplexeOperation, 68, "testDotProduct" ) {}
 void runTest() { suite_TensorInlineComplexeOperation.testDotProduct(); }
} testDescription_suite_TensorInlineComplexeOperation_testDotProduct;

#include <cxxtest/Root.cpp>
const char* CxxTest::RealWorldDescription::_worldName = "cxxtest";
