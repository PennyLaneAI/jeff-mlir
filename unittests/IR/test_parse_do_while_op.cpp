#include "jeff/IR/JeffDialect.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LogicalResult.h>

#include <string>

using namespace mlir;

class DoWhileOpTest : public ::testing::Test {
  protected:
    MLIRContext ctx;
    ScopedDiagnosticHandler handler{&ctx, [](Diagnostic&) { return success(); }};

    void SetUp() override { ctx.loadDialect<jeff::JeffDialect, func::FuncDialect>(); }
};

// Jeff SCF regions are isolated from above: every value used inside a region must come
// from a block argument or be computed locally.
// Tests with carried values pass the loop predicate through as an additional in-value
// (idiomatic Jeff).
// The `NoArgs` and `ExplicitEmptyBodyYield` tests exercise the no-in-values parser path
// and therefore compute the predicate inside the condition via `jeff.int_const1`,
// since they have no operands to inherit one from.

// === Valid tests ===

TEST_F(DoWhileOpTest, NoArgs) {
    const std::string src = R"MLIR(
      func.func @f() {
        jeff.doWhile args() {
        } args() {
          %c_pred = jeff.int_const1(true) : i1
          jeff.yield %c_pred : i1
        }
        return
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(DoWhileOpTest, WithArgsSingle) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %pred: i1) -> i32 {
        %r1, %r2 = jeff.doWhile : (i32, i1) args(%b_x = %a, %b_pred = %pred) {
          jeff.yield %b_x, %b_pred : i32, i1
        } args(%c_x, %c_pred) {
          jeff.yield %c_pred : i1
        }
        return %r1 : i32
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(DoWhileOpTest, WithArgsMultiple) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %b: i64, %pred: i1) -> (i32, i64) {
        %r1, %r2, %r3 = jeff.doWhile : (i32, i64, i1) args(%b_x = %a, %b_y = %b, %b_pred = %pred) {
          jeff.yield %b_x, %b_y, %b_pred : i32, i64, i1
        } args(%c_x, %c_y, %c_pred) {
          jeff.yield %c_pred : i1
        }
        return %r1, %r2 : i32, i64
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(DoWhileOpTest, Nested) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %pred: i1) -> i32 {
        %r1, %r2 = jeff.doWhile : (i32, i1) args(%b_x = %a, %b_pred = %pred) {
          %s1, %s2 = jeff.doWhile : (i32, i1) args(%bb = %b_x, %bbp = %b_pred) {
            jeff.yield %bb, %bbp : i32, i1
          } args(%cc, %ccp) {
            jeff.yield %ccp : i1
          }
          jeff.yield %s1, %s2 : i32, i1
        } args(%c_x, %c_pred) {
          jeff.yield %c_pred : i1
        }
        return %r1 : i32
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

// `DoWhileOp::print` elides the empty yield for the body when there are no in-values,
// but bare `jeff.yield` is still valid input. It can come from `YieldOp`'s own printer,
// generic-form output (`-mlir-print-op-generic`), or handwritten MLIR.
// The parser must accept this shape.
TEST_F(DoWhileOpTest, ExplicitEmptyBodyYield) {
    const std::string src = R"MLIR(
      func.func @f() {
        jeff.doWhile args() {
          jeff.yield
        } args() {
          %c_pred = jeff.int_const1(true) : i1
          jeff.yield %c_pred : i1
        }
        return
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

// Parse → print → parse → print, then assert idempotent.
// The first round normalizes (whitespace, SSA names).
// The second round must be a no-op.
TEST_F(DoWhileOpTest, RoundTripIdempotent) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %b: i64, %pred: i1) -> (i32, i64) {
        %r1, %r2, %r3 = jeff.doWhile : (i32, i64, i1) args(%b_x = %a, %b_y = %b, %b_pred = %pred) {
          jeff.yield %b_x, %b_y, %b_pred : i32, i64, i1
        } args(%c_x, %c_y, %c_pred) {
          jeff.yield %c_pred : i1
        }
        return %r1, %r2 : i32, i64
      }
    )MLIR";

    const auto module1 = parseSourceString<ModuleOp>(src, &ctx);
    ASSERT_TRUE(module1);
    std::string printed1;
    llvm::raw_string_ostream(printed1) << *module1;

    const auto module2 = parseSourceString<ModuleOp>(printed1, &ctx);
    ASSERT_TRUE(module2);
    std::string printed2;
    llvm::raw_string_ostream(printed2) << *module2;

    EXPECT_EQ(printed1, printed2);
}

// === Invalid syntax tests (parse-level) ===

TEST_F(DoWhileOpTest, InvalidMissingArgsKeyword) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %pred: i1) {
        jeff.doWhile : (i32, i1) (%b_x = %a, %b_pred = %pred) {
          jeff.yield %b_x, %b_pred : i32, i1
        } args(%c_x, %c_pred) {
          jeff.yield %c_pred : i1
        }
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

// In-values count and body args count differ.
TEST_F(DoWhileOpTest, InvalidArgCountMismatchWithTypes) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %pred: i1) {
        jeff.doWhile : (i32, i1, i64) args(%b_x = %a, %b_pred = %pred) {
          jeff.yield %b_x, %b_pred : i32, i1
        } args(%c_x, %c_pred) {
          jeff.yield %c_pred : i1
        }
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

// Body args count and condition args count differ.
TEST_F(DoWhileOpTest, InvalidBodyCondArgCountMismatch) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %b: i64, %pred: i1) {
        jeff.doWhile : (i32, i64, i1) args(%b_x = %a, %b_y = %b, %b_pred = %pred) {
          jeff.yield %b_x, %b_y, %b_pred : i32, i64, i1
        } args(%c_x, %c_pred) {
          jeff.yield %c_pred : i1
        }
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

// Missing in-value types with non-empty body args.
TEST_F(DoWhileOpTest, InvalidMissingTypeAnnotation) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %pred: i1) {
        jeff.doWhile args(%b_x = %a, %b_pred = %pred) {
          jeff.yield %b_x, %b_pred : i32, i1
        } args(%c_x, %c_pred) {
          jeff.yield %c_pred : i1
        }
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}
