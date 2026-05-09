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

class ForOpTest : public ::testing::Test {
  protected:
    MLIRContext ctx;
    ScopedDiagnosticHandler handler{&ctx, [](Diagnostic&) { return success(); }};

    void SetUp() override { ctx.loadDialect<jeff::JeffDialect, func::FuncDialect>(); }
};

// === Valid tests ===

TEST_F(ForOpTest, BasicFormI32) {
    const auto module = parseSourceString<ModuleOp>(R"MLIR(
      func.func @f(%lo: i32, %hi: i32, %s: i32) {
        jeff.for %i = %lo to %hi step %s : i32 {}
        return
      }
  )MLIR",
                                                    &ctx);
    ASSERT_TRUE(module);
}

TEST_F(ForOpTest, BasicFormI64) {
    const auto module = parseSourceString<ModuleOp>(R"MLIR(
      func.func @f(%lo: i64, %hi: i64, %s: i64) {
        jeff.for %i = %lo to %hi step %s : i64 {}
        return
      }
  )MLIR",
                                                    &ctx);
    ASSERT_TRUE(module);
}

// Body has no explicit `jeff.yield`.
// `SingleBlockImplicitTerminator` should auto-insert one
// via `ForOp::ensureTerminator`.
TEST_F(ForOpTest, ImplicitYield) {
    const auto module = parseSourceString<ModuleOp>(R"MLIR(
      func.func @f(%lo: i32, %hi: i32, %s: i32) {
        jeff.for %i = %lo to %hi step %s : i32 {}
        return
      }
  )MLIR",
                                                    &ctx);
    ASSERT_TRUE(module);
}

TEST_F(ForOpTest, WithArgsSingle) {
    const auto module = parseSourceString<ModuleOp>(R"MLIR(
      func.func @f(%lo: i32, %hi: i32, %s: i32, %init: i32) -> i32 {
        %r = jeff.for %i = %lo to %hi step %s args(%acc = %init) -> (i32) : i32 {
          jeff.yield %acc : i32
        }
        return %r : i32
      }
  )MLIR",
                                                    &ctx);
    ASSERT_TRUE(module);
}

TEST_F(ForOpTest, WithArgsMultiple) {
    const auto module = parseSourceString<ModuleOp>(R"MLIR(
      func.func @f(%lo: i32, %hi: i32, %s: i32, %a: i32, %b: i64) -> (i32, i64) {
        %r1, %r2 = jeff.for %i = %lo to %hi step %s args(%x = %a, %y = %b) -> (i32, i64) : i32 {
          jeff.yield %x, %y : i32, i64
        }
        return %r1, %r2 : i32, i64
      }
  )MLIR",
                                                    &ctx);
    ASSERT_TRUE(module);
}

TEST_F(ForOpTest, Nested) {
    const auto module = parseSourceString<ModuleOp>(R"MLIR(
      func.func @f(%lo: i32, %hi: i32, %s: i32) {
        jeff.for %i = %lo to %hi step %s : i32 {
          jeff.for %j = %lo to %hi step %s : i32 {}
        }
        return
      }
  )MLIR",
                                                    &ctx);
    ASSERT_TRUE(module);
}

// `ForOp::print` elides the empty yield,
// but bare `jeff.yield` is still valid input.
// It can come from `YieldOp`'s own printer,
// generic-form output (`-mlir-print-op-generic`), or
// handwritten MLIR.
// The parser must accept this shape.
TEST_F(ForOpTest, ExplicitEmptyYield) {
    const auto module = parseSourceString<ModuleOp>(R"MLIR(
      func.func @f(%lo: i32, %hi: i32, %s: i32) {
        jeff.for %i = %lo to %hi step %s : i32 {
          jeff.yield
        }
        return
      }
  )MLIR",
                                                    &ctx);
    ASSERT_TRUE(module);
}

// Parse → print → parse → print, then assert idempotent.
// The first round normalizes (whitespace, SSA names).
// The second round must be a no-op.
TEST_F(ForOpTest, RoundTripIdempotent) {
    const std::string src = R"MLIR(
      func.func @f(%lo: i32, %hi: i32, %s: i32, %init: i32) -> i32 {
        %r = jeff.for %i = %lo to %hi step %s args(%acc = %init) -> (i32) : i32 {
          jeff.yield %acc : i32
        }
        return %r : i32
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

TEST_F(ForOpTest, InvalidMissingEquals) {
    const auto module = parseSourceString<ModuleOp>(R"MLIR(
      func.func @f(%lo: i32, %hi: i32, %s: i32) {
        jeff.for %i %lo to %hi step %s : i32 {}
        return
      }
  )MLIR",
                                                    &ctx);
    ASSERT_FALSE(module);
}

TEST_F(ForOpTest, InvalidMissingTo) {
    const auto module = parseSourceString<ModuleOp>(R"MLIR(
      func.func @f(%lo: i32, %hi: i32, %s: i32) {
        jeff.for %i = %lo %hi step %s : i32 {}
        return
      }
  )MLIR",
                                                    &ctx);
    ASSERT_FALSE(module);
}

TEST_F(ForOpTest, InvalidMissingType) {
    const auto module = parseSourceString<ModuleOp>(R"MLIR(
      func.func @f(%lo: i32, %hi: i32, %s: i32) {
        jeff.for %i = %lo to %hi step %s {}
        return
      }
  )MLIR",
                                                    &ctx);
    ASSERT_FALSE(module);
}

TEST_F(ForOpTest, InvalidArgsWithoutArrow) {
    const auto module = parseSourceString<ModuleOp>(R"MLIR(
      func.func @f(%lo: i32, %hi: i32, %s: i32, %init: i32) {
        jeff.for %i = %lo to %hi step %s args(%acc = %init) : i32 {
          jeff.yield %acc : i32
        }
        return
      }
  )MLIR",
                                                    &ctx);
    ASSERT_FALSE(module);
}

// 2 region args, 1 result type.
// Caught by the explicit size check in `parse`.
TEST_F(ForOpTest, InvalidArgCountMismatch) {
    const auto module = parseSourceString<ModuleOp>(R"MLIR(
      func.func @f(%lo: i32, %hi: i32, %s: i32, %x: i32, %y: i32) {
        jeff.for %i = %lo to %hi step %s args(%a = %x, %b = %y) -> (i32) : i32 {
          jeff.yield %a : i32
        }
        return
      }
  )MLIR",
                                                    &ctx);
    ASSERT_FALSE(module);
}

// === Invalid semantics tests (verify-level) ===

// `index` is rejected by SupportedIntType.
TEST_F(ForOpTest, InvalidIndexType) {
    const auto module = parseSourceString<ModuleOp>(R"MLIR(
      func.func @f(%lo: index, %hi: index, %s: index) {
        jeff.for %i = %lo to %hi step %s : index {}
        return
      }
  )MLIR",
                                                    &ctx);
    ASSERT_FALSE(module);
}

// Floating-point types are rejected by SupportedIntType.
TEST_F(ForOpTest, InvalidFloatType) {
    const auto module = parseSourceString<ModuleOp>(R"MLIR(
      func.func @f(%lo: f32, %hi: f32, %s: f32) {
        jeff.for %i = %lo to %hi step %s : f32 {}
        return
      }
  )MLIR",
                                                    &ctx);
    ASSERT_FALSE(module);
}
