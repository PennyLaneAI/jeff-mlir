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

class SwitchOpTest : public ::testing::Test {
  protected:
    MLIRContext ctx;
    ScopedDiagnosticHandler handler{&ctx, [](Diagnostic&) { return success(); }};

    void SetUp() override { ctx.loadDialect<jeff::JeffDialect, func::FuncDialect>(); }
};

// === Valid tests ===

// Structurally legal: a switch with zero branches and no default.
// (See "Underspecified" in the op description.)
TEST_F(SwitchOpTest, ZeroCasesAndNoDefault) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32) {
        jeff.switch (%sel) : (i32) -> ()
        return
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(SwitchOpTest, NoInValuesNoResults) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32) {
        jeff.switch (%sel) : (i32) -> ()
        case 0 args() {}
        case 1 args() {}
        default args() {}
        return
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(SwitchOpTest, WithInValuesAndResults) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32, %a: i32, %b: i64) -> (i32, i64) {
        %r1, %r2 = jeff.switch (%sel, %a, %b) : (i32, i32, i64) -> (i32, i64)
        case 0 args(%x, %y) {
          jeff.yield %x, %y : i32, i64
        }
        case 1 args(%x, %y) {
          jeff.yield %x, %y : i32, i64
        }
        default args(%x, %y) {
          jeff.yield %x, %y : i32, i64
        }
        return %r1, %r2 : i32, i64
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(SwitchOpTest, NoDefault) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32, %a: i32) -> i32 {
        %r = jeff.switch (%sel, %a) : (i32, i32) -> (i32)
        case 0 args(%x) {
          jeff.yield %x : i32
        }
        case 1 args(%x) {
          jeff.yield %x : i32
        }
        return %r : i32
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(SwitchOpTest, OnlyDefault) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32, %a: i32) -> i32 {
        %r = jeff.switch (%sel, %a) : (i32, i32) -> (i32)
        default args(%x) {
          jeff.yield %x : i32
        }
        return %r : i32
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

// In-value types and result types are independent:
// the in-values are (i32, i64), but the op yields a single i1.
TEST_F(SwitchOpTest, DecoupledInValueAndResultTypes) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32, %a: i32, %b: i64, %p: i1) -> i1 {
        %r = jeff.switch (%sel, %a, %b) : (i32, i32, i64) -> (i1)
        case 0 args(%x, %y) {
          jeff.yield %p : i1
        }
        default args(%x, %y) {
          jeff.yield %p : i1
        }
        return %r : i1
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(SwitchOpTest, Nested) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32, %a: i32) -> i32 {
        %r = jeff.switch (%sel, %a) : (i32, i32) -> (i32)
        case 0 args(%x) {
          %s = jeff.switch (%sel, %x) : (i32, i32) -> (i32)
          case 0 args(%xx) {
            jeff.yield %xx : i32
          }
          default args(%xx) {
            jeff.yield %xx : i32
          }
          jeff.yield %s : i32
        }
        default args(%x) {
          jeff.yield %x : i32
        }
        return %r : i32
      }
    )MLIR";
    ASSERT_TRUE(parseSourceString<ModuleOp>(src, &ctx));
}

// Parse → print → parse → print, then assert idempotent.
TEST_F(SwitchOpTest, RoundTripIdempotent) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32, %a: i32, %b: i64) -> (i32, i64) {
        %r1, %r2 = jeff.switch (%sel, %a, %b) : (i32, i32, i64) -> (i32, i64)
        case 0 args(%x, %y) {
          jeff.yield %x, %y : i32, i64
        }
        case 1 args(%x, %y) {
          jeff.yield %x, %y : i32, i64
        }
        default args(%x, %y) {
          jeff.yield %x, %y : i32, i64
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

// Case labels must start at 0 and increase by 1.
TEST_F(SwitchOpTest, InvalidNonContiguousCaseLabels) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32) {
        jeff.switch (%sel) : (i32) -> ()
        case 0 args() {}
        case 2 args() {}
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(SwitchOpTest, InvalidCaseLabelNotStartingAtZero) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32) {
        jeff.switch (%sel) : (i32) -> ()
        case 1 args() {}
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(SwitchOpTest, InvalidMissingArgsKeyword) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32, %a: i32) {
        jeff.switch (%sel, %a) : (i32, i32) -> ()
        case 0 (%x) {}
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

// Region arg count must match the number of in-values.
TEST_F(SwitchOpTest, InvalidRegionArgCountMismatch) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32, %a: i32, %b: i64) {
        jeff.switch (%sel, %a, %b) : (i32, i32, i64) -> ()
        case 0 args(%x) {}
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

// Operand-type count must match operand count.
TEST_F(SwitchOpTest, InvalidOperandTypeCountMismatch) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32, %a: i32) {
        jeff.switch (%sel, %a) : (i32) -> ()
        case 0 args(%x) {}
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

// The selector must be a `SupportedIntType` (i1/i8/i16/i32/i64).
// `index` is rejected by the ODS-generated verifier.
TEST_F(SwitchOpTest, InvalidSelectorTypeIndex) {
    const std::string src = R"MLIR(
      func.func @f(%sel: index) {
        jeff.switch (%sel) : (index) -> ()
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

TEST_F(SwitchOpTest, InvalidSelectorTypeFloat) {
    const std::string src = R"MLIR(
      func.func @f(%sel: f32) {
        jeff.switch (%sel) : (f32) -> ()
        return
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

// === Invalid verifier tests ===

// Yield operand count must match the op's result count.
TEST_F(SwitchOpTest, InvalidYieldCountMismatch) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32, %a: i32) -> i32 {
        %r = jeff.switch (%sel, %a) : (i32, i32) -> (i32)
        case 0 args(%x) {
          jeff.yield
        }
        default args(%x) {
          jeff.yield %x : i32
        }
        return %r : i32
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}

// Yield operand type must match the op's result type.
TEST_F(SwitchOpTest, InvalidYieldTypeMismatch) {
    const std::string src = R"MLIR(
      func.func @f(%sel: i32, %a: i32, %b: i64) -> i32 {
        %r = jeff.switch (%sel, %a, %b) : (i32, i32, i64) -> (i32)
        case 0 args(%x, %y) {
          jeff.yield %y : i64
        }
        default args(%x, %y) {
          jeff.yield %x : i32
        }
        return %r : i32
      }
    )MLIR";
    ASSERT_FALSE(parseSourceString<ModuleOp>(src, &ctx));
}
