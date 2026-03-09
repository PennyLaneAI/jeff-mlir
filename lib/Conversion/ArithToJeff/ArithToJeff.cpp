/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "jeff/Conversion/ArithToJeff/ArithToJeff.h"

#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numbers>
#include <string>
#include <utility>

namespace mlir {

#define GEN_PASS_DEF_ARITHTOJEFF
#include "jeff/Conversion/ArithToJeff/ArithToJeff.h.inc"

/**
 * @brief Converts arith.constant to Jeff
 *
 * @par Example:
 * ```mlir
 * %0 = arith.constant 0 : i64
 * ```
 * is converted to
 * ```mlir
 * %0 = jeff.int_const64(0) : i64
 * ```
 */
struct ConvertArithConstOpToJeff final : OpConversionPattern<arith::ConstantOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
        auto value = op.getValue();
        return llvm::TypeSwitch<Type, LogicalResult>(op.getType())
            .Case<FloatType>([&](auto type) -> LogicalResult {
                auto floatAttr = llvm::dyn_cast<FloatAttr>(value);
                if (!floatAttr) {
                    return rewriter.notifyMatchFailure(op, "Expected float attribute");
                }
                switch (type.getWidth()) {
                case 64:
                    rewriter.replaceOpWithNewOp<jeff::FloatConst64Op>(op, floatAttr);
                    return success();
                default:
                    return rewriter.notifyMatchFailure(op, "Unsupported type");
                }
            })
            .Case<IntegerType>([&](auto type) -> LogicalResult {
                auto intAttr = llvm::dyn_cast<IntegerAttr>(value);
                if (!intAttr) {
                    return rewriter.notifyMatchFailure(op, "Expected integer attribute");
                }
                switch (type.getWidth()) {
                case 1:
                    rewriter.replaceOpWithNewOp<jeff::IntConst1Op>(op, intAttr);
                    return success();
                case 64:
                    rewriter.replaceOpWithNewOp<jeff::IntConst64Op>(op, intAttr);
                    return success();
                default:
                    return rewriter.notifyMatchFailure(op, "Unsupported type");
                }
            })
            .Default([&](auto) -> LogicalResult {
                return rewriter.notifyMatchFailure(op, "Unsupported type");
            });
    }
};

/**
 * @brief Type converter for `arith`-to-Jeff conversion
 */
class ArithToJeffTypeConverter final : public TypeConverter {
  public:
    explicit ArithToJeffTypeConverter(MLIRContext* ctx) {
        // Identity conversion for all types by default
        addConversion([](Type type) { return type; });
    }
};

/**
 * @brief Pass for converting `arith` operations to Jeff operations
 */
struct ArithToJeff final : impl::ArithToJeffBase<ArithToJeff> {
    using ArithToJeffBase::ArithToJeffBase;

  protected:
    void runOnOperation() override {
        MLIRContext* context = &getContext();
        auto* module = getOperation();

        ConversionTarget target(*context);
        RewritePatternSet patterns(context);
        ArithToJeffTypeConverter typeConverter(context);

        // Configure conversion target
        target.addIllegalDialect<arith::ArithDialect>();
        target.addLegalDialect<jeff::JeffDialect>();

        // Register operation conversion patterns
        patterns.add<ConvertArithConstOpToJeff>(typeConverter, context);

        // Apply the conversion
        if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
            signalPassFailure();
        }
    }
};

} // namespace mlir
