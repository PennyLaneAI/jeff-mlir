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

class WhileOpTest : public ::testing::Test {
  protected:
    MLIRContext ctx;
    ScopedDiagnosticHandler handler{&ctx, [](Diagnostic&) { return success(); }};

    void SetUp() override { ctx.loadDialect<jeff::JeffDialect, func::FuncDialect>(); }
};

// === Valid tests ===

TEST_F(WhileOpTest, NoArgs) {
    const std::string src = R"MLIR(
      func.func @f(%pred: i1) {
        jeff.while args() {
          jeff.yield %pred : i1
        } args() {
        }
        return
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(WhileOpTest, WithArgsSingle) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %pred: i1) -> i32 {
        %r = jeff.while : (i32) args(%c_x = %a) {
          jeff.yield %pred : i1
        } args(%b_x) {
          jeff.yield %b_x : i32
        }
        return %r : i32
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(WhileOpTest, WithArgsMultiple) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %b: i64, %pred: i1) -> (i32, i64) {
        %r1, %r2 = jeff.while : (i32, i64) args(%c_x = %a, %c_y = %b) {
          jeff.yield %pred : i1
        } args(%b_x, %b_y) {
          jeff.yield %b_x, %b_y : i32, i64
        }
        return %r1, %r2 : i32, i64
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(WhileOpTest, Nested) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %pred: i1) -> i32 {
        %r = jeff.while : (i32) args(%c_x = %a) {
          jeff.yield %pred : i1
        } args(%b_x) {
          %s = jeff.while : (i32) args(%cc = %b_x) {
            jeff.yield %pred : i1
          } args(%bb) {
            jeff.yield %bb : i32
          }
          jeff.yield %s : i32
        }
        return %r : i32
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

// `WhileOp::print` elides the empty yield for the body when there are no
// in-values, but bare `jeff.yield` is still valid input. It can come from
// `YieldOp`'s own printer, generic-form output (`-mlir-print-op-generic`),
// or handwritten MLIR. The parser must accept this shape.
TEST_F(WhileOpTest, ExplicitEmptyBodyYield) {
    const std::string src = R"MLIR(
      func.func @f(%pred: i1) {
        jeff.while args() {
          jeff.yield %pred : i1
        } args() {
          jeff.yield
        }
        return
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

// Parse → print → parse → print, then assert idempotent.
// The first round normalizes (whitespace, SSA names).
// The second round must be a no-op.
TEST_F(WhileOpTest, RoundTripIdempotent) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %b: i64, %pred: i1) -> (i32, i64) {
        %r1, %r2 = jeff.while : (i32, i64) args(%c_x = %a, %c_y = %b) {
          jeff.yield %pred : i1
        } args(%b_x, %b_y) {
          jeff.yield %b_x, %b_y : i32, i64
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

TEST_F(WhileOpTest, InvalidMissingArgsKeyword) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %pred: i1) {
        jeff.while : (i32) (%c_x = %a) {
          jeff.yield %pred : i1
        } args(%b_x) {
          jeff.yield %b_x : i32
        }
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

// `: (...)` is present but `args(...)` is empty (or vice versa).
TEST_F(WhileOpTest, InvalidArgCountMismatchWithTypes) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %pred: i1) {
        jeff.while : (i32, i64) args(%c_x = %a) {
          jeff.yield %pred : i1
        } args(%b_x) {
          jeff.yield %b_x : i32
        }
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

// Condition has 2 args but body has 1.
TEST_F(WhileOpTest, InvalidCondBodyArgCountMismatch) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %b: i64, %pred: i1) {
        jeff.while : (i32, i64) args(%c_x = %a, %c_y = %b) {
          jeff.yield %pred : i1
        } args(%b_x) {
          jeff.yield %b_x : i32
        }
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

// `args(...)` is non-empty but no `: (...)` is provided.
TEST_F(WhileOpTest, InvalidMissingTypeAnnotation) {
    const std::string src = R"MLIR(
      func.func @f(%a: i32, %pred: i1) {
        jeff.while args(%c_x = %a) {
          jeff.yield %pred : i1
        } args(%b_x) {
          jeff.yield %b_x : i32
        }
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}
