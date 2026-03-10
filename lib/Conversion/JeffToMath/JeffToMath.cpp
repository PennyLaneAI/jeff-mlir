#include "jeff/Conversion/JeffToMath/JeffToMath.h"

#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffOps.h"

#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {

#define GEN_PASS_DEF_JEFFTOMATH
#include "jeff/Conversion/JeffToMath/JeffToMath.h.inc"

struct ConvertIntUnaryOpToMath final : OpConversionPattern<jeff::IntUnaryOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::IntUnaryOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        switch (op.getOp()) {
        case jeff::IntUnaryOperation::_abs:
            rewriter.replaceOpWithNewOp<math::AbsIOp>(op, a);
            break;
        default:
            rewriter.notifyMatchFailure(op, "Unknown unary operation");
        }
        return success();
    }
};

struct ConvertJeffIntBinaryOpToMath final : OpConversionPattern<jeff::IntBinaryOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::IntBinaryOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        auto b = adaptor.getB();
        switch (op.getOp()) {
        case jeff::IntBinaryOperation::_pow:
            rewriter.replaceOpWithNewOp<math::IPowIOp>(op, a, b);
            break;
        default:
            rewriter.notifyMatchFailure(op, "Unknown binary operation");
        }
        return success();
    }
};

struct ConvertJeffFloatUnaryOpToMath final : OpConversionPattern<jeff::FloatUnaryOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::FloatUnaryOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        switch (op.getOp()) {
        case jeff::FloatUnaryOperation::_sqrt:
            rewriter.replaceOpWithNewOp<math::SqrtOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_abs:
            rewriter.replaceOpWithNewOp<math::AbsFOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_ceil:
            rewriter.replaceOpWithNewOp<math::CeilOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_floor:
            rewriter.replaceOpWithNewOp<math::FloorOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_exp:
            rewriter.replaceOpWithNewOp<math::ExpOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_log:
            rewriter.replaceOpWithNewOp<math::LogOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_sin:
            rewriter.replaceOpWithNewOp<math::SinOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_cos:
            rewriter.replaceOpWithNewOp<math::CosOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_tan:
            rewriter.replaceOpWithNewOp<math::TanOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_asin:
            rewriter.replaceOpWithNewOp<math::AsinOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_acos:
            rewriter.replaceOpWithNewOp<math::AcosOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_atan:
            rewriter.replaceOpWithNewOp<math::AtanOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_sinh:
            rewriter.replaceOpWithNewOp<math::SinhOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_cosh:
            rewriter.replaceOpWithNewOp<math::CoshOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_tanh:
            rewriter.replaceOpWithNewOp<math::TanhOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_asinh:
            rewriter.replaceOpWithNewOp<math::AsinhOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_acosh:
            rewriter.replaceOpWithNewOp<math::AcoshOp>(op, a);
            break;
        case jeff::FloatUnaryOperation::_atanh:
            rewriter.replaceOpWithNewOp<math::AtanhOp>(op, a);
            break;
        default:
            rewriter.notifyMatchFailure(op, "Unknown unary operation");
        }
        return success();
    }
};

struct ConvertJeffFloatBinaryOpToMath final : OpConversionPattern<jeff::FloatBinaryOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::FloatBinaryOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        auto b = adaptor.getB();
        switch (op.getOp()) {
        case jeff::FloatBinaryOperation::_atan2:
            rewriter.replaceOpWithNewOp<math::Atan2Op>(op, a, b);
            break;
        case jeff::FloatBinaryOperation::_pow:
            rewriter.replaceOpWithNewOp<math::PowFOp>(op, a, b);
            break;
        default:
            rewriter.notifyMatchFailure(op, "Unknown binary operation");
        }
        return success();
    }
};

struct ConvertJeffFloatIsOpToMath final : OpConversionPattern<jeff::FloatIsOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::FloatIsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        switch (op.getOp()) {
        case jeff::FloatIsOperation::_isNan:
            rewriter.replaceOpWithNewOp<math::IsNaNOp>(op, a);
            break;
        case jeff::FloatIsOperation::_isInf:
            rewriter.replaceOpWithNewOp<math::IsInfOp>(op, a);
            break;
        default:
            rewriter.notifyMatchFailure(op, "Unknown binary operation");
        }
        return success();
    }
};

/**
 * @brief Pass for converting Jeff operations to `math` operations
 */
struct JeffToMath final : impl::JeffToMathBase<JeffToMath> {
    using JeffToMathBase::JeffToMathBase;

  protected:
    void runOnOperation() override {
        MLIRContext* context = &getContext();
        auto* module = getOperation();

        ConversionTarget target(*context);
        target.addIllegalOp<jeff::FloatUnaryOp, jeff::FloatIsOp>();
        target.addDynamicallyLegalOp<jeff::IntUnaryOp>([](auto* op) {
            auto intOp = llvm::cast<jeff::IntUnaryOp>(op);
            if (intOp.getOp() == jeff::IntUnaryOperation::_abs) {
                return false;
            }
            return true;
        });
        target.addDynamicallyLegalOp<jeff::IntBinaryOp>([](auto* op) {
            auto intOp = llvm::cast<jeff::IntBinaryOp>(op);
            if (intOp.getOp() == jeff::IntBinaryOperation::_pow) {
                return false;
            }
            return true;
        });
        target.addDynamicallyLegalOp<jeff::FloatBinaryOp>([](auto* op) {
            auto floatOp = llvm::cast<jeff::FloatBinaryOp>(op);
            auto binaryOp = floatOp.getOp();
            if (binaryOp == jeff::FloatBinaryOperation::_pow ||
                binaryOp == jeff::FloatBinaryOperation::_atan2) {
                return false;
            }
            return true;
        });
        target.addLegalDialect<math::MathDialect>();

        RewritePatternSet patterns(context);
        patterns.add<ConvertIntUnaryOpToMath, ConvertJeffIntBinaryOpToMath,
                     ConvertJeffFloatUnaryOpToMath, ConvertJeffFloatBinaryOpToMath,
                     ConvertJeffFloatIsOpToMath>(context);

        if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
            signalPassFailure();
        }
    }
};

} // namespace mlir
