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

// === Valid tests ===

TEST_F(DoWhileOpTest, NoArgs) {
    const std::string src = R"MLIR(
      func.func @f(%pred: i1) {
        jeff.doWhile args() {
        } args() {
          jeff.yield %pred : i1
        }
        return
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(DoWhileOpTest, WithArgsSingle) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %pred: i1) -> i32 {
        %r = jeff.doWhile : (i32) args(%b_x = %a) {
          jeff.yield %b_x : i32
        } args(%c_x) {
          jeff.yield %pred : i1
        }
        return %r : i32
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(DoWhileOpTest, WithArgsMultiple) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %b: i64, %pred: i1) -> (i32, i64) {
        %r1, %r2 = jeff.doWhile : (i32, i64) args(%b_x = %a, %b_y = %b) {
          jeff.yield %b_x, %b_y : i32, i64
        } args(%c_x, %c_y) {
          jeff.yield %pred : i1
        }
        return %r1, %r2 : i32, i64
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(DoWhileOpTest, Nested) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %pred: i1) -> i32 {
        %r = jeff.doWhile : (i32) args(%b_x = %a) {
          %s = jeff.doWhile : (i32) args(%bb = %b_x) {
            jeff.yield %bb : i32
          } args(%cc) {
            jeff.yield %pred : i1
          }
          jeff.yield %s : i32
        } args(%c_x) {
          jeff.yield %pred : i1
        }
        return %r : i32
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

// `DoWhileOp::print` elides the empty yield for the body when there are no
// in-values, but bare `jeff.yield` is still valid input. It can come from
// `YieldOp`'s own printer, generic-form output (`-mlir-print-op-generic`),
// or handwritten MLIR. The parser must accept this shape.
TEST_F(DoWhileOpTest, ExplicitEmptyBodyYield) {
    const std::string src = R"MLIR(
      func.func @f(%pred: i1) {
        jeff.doWhile args() {
          jeff.yield
        } args() {
          jeff.yield %pred : i1
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
        %r1, %r2 = jeff.doWhile : (i32, i64) args(%b_x = %a, %b_y = %b) {
          jeff.yield %b_x, %b_y : i32, i64
        } args(%c_x, %c_y) {
          jeff.yield %pred : i1
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
        jeff.doWhile : (i32) (%b_x = %a) {
          jeff.yield %b_x : i32
        } args(%c_x) {
          jeff.yield %pred : i1
        }
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

// `: (...)` is present but `args(...)` is empty (or vice versa).
TEST_F(DoWhileOpTest, InvalidArgCountMismatchWithTypes) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %pred: i1) {
        jeff.doWhile : (i32, i64) args(%b_x = %a) {
          jeff.yield %b_x : i32
        } args(%c_x) {
          jeff.yield %pred : i1
        }
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

// Body has 2 args but condition has 1.
TEST_F(DoWhileOpTest, InvalidBodyCondArgCountMismatch) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %b: i64, %pred: i1) {
        jeff.doWhile : (i32, i64) args(%b_x = %a, %b_y = %b) {
          jeff.yield %b_x, %b_y : i32, i64
        } args(%c_x) {
          jeff.yield %pred : i1
        }
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

// `args(...)` is non-empty but no `: (...)` is provided.
TEST_F(DoWhileOpTest, InvalidMissingTypeAnnotation) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %pred: i1) {
        jeff.doWhile args(%b_x = %a) {
          jeff.yield %b_x : i32
        } args(%c_x) {
          jeff.yield %pred : i1
        }
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}
