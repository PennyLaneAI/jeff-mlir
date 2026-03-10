#include "jeff/Conversion/MathToJeff/MathToJeff.h"

#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Math/IR/Math.h>
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

#define GEN_PASS_DEF_MATHTOJEFF
#include "jeff/Conversion/MathToJeff/MathToJeff.h.inc"

struct ConvertMathAbsIOpToJeff final : OpConversionPattern<math::AbsIOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(math::AbsIOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::IntUnaryOp>(op, adaptor.getOperand(),
                                                      jeff::IntUnaryOperation::_abs);
        return success();
    }
};

struct ConvertMathIPowIOpToJeff final : OpConversionPattern<math::IPowIOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(math::IPowIOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::IntBinaryOp>(op, adaptor.getLhs(), adaptor.getRhs(),
                                                       jeff::IntBinaryOperation::_pow);
        return success();
    }
};

template <typename MathOp, jeff::FloatUnaryOperation JeffOp>
struct ConvertMathFloatUnaryOp final : OpConversionPattern<MathOp> {
    using OpConversionPattern<MathOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(MathOp op, typename MathOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::FloatUnaryOp>(op, adaptor.getOperand(), JeffOp);
        return success();
    }
};

template <typename MathOp, jeff::FloatBinaryOperation JeffOp>
struct ConvertMathFloatBinaryOp final : OpConversionPattern<MathOp> {
    using OpConversionPattern<MathOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(MathOp op, typename MathOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::FloatBinaryOp>(op, adaptor.getLhs(), adaptor.getRhs(),
                                                         JeffOp);
        return success();
    }
};

template <typename MathOp, jeff::FloatIsOperation JeffOp>
struct ConvertMathFloatIsOp final : OpConversionPattern<MathOp> {
    using OpConversionPattern<MathOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(MathOp op, typename MathOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::FloatIsOp>(op, adaptor.getOperand(), JeffOp);
        return success();
    }
};

/**
 * @brief Pass for converting `math` operations to Jeff operations
 */
struct MathToJeff final : impl::MathToJeffBase<MathToJeff> {
    using MathToJeffBase::MathToJeffBase;

  protected:
    void runOnOperation() override {
        MLIRContext* context = &getContext();
        auto* module = getOperation();

        ConversionTarget target(*context);
        target.addLegalDialect<jeff::JeffDialect>();
        target.addIllegalDialect<math::MathDialect>();

        RewritePatternSet patterns(context);
        patterns.add<ConvertMathAbsIOpToJeff, ConvertMathIPowIOpToJeff,
                     ConvertMathFloatUnaryOp<math::SqrtOp, jeff::FloatUnaryOperation::_sqrt>,
                     ConvertMathFloatUnaryOp<math::AbsFOp, jeff::FloatUnaryOperation::_abs>,
                     ConvertMathFloatUnaryOp<math::CeilOp, jeff::FloatUnaryOperation::_ceil>,
                     ConvertMathFloatUnaryOp<math::FloorOp, jeff::FloatUnaryOperation::_floor>,
                     ConvertMathFloatUnaryOp<math::ExpOp, jeff::FloatUnaryOperation::_exp>,
                     ConvertMathFloatUnaryOp<math::LogOp, jeff::FloatUnaryOperation::_log>,
                     ConvertMathFloatUnaryOp<math::SinOp, jeff::FloatUnaryOperation::_sin>,
                     ConvertMathFloatUnaryOp<math::CosOp, jeff::FloatUnaryOperation::_cos>,
                     ConvertMathFloatUnaryOp<math::TanOp, jeff::FloatUnaryOperation::_tan>,
                     ConvertMathFloatUnaryOp<math::AsinOp, jeff::FloatUnaryOperation::_asin>,
                     ConvertMathFloatUnaryOp<math::AcosOp, jeff::FloatUnaryOperation::_acos>,
                     ConvertMathFloatUnaryOp<math::AtanOp, jeff::FloatUnaryOperation::_atan>,
                     ConvertMathFloatUnaryOp<math::SinhOp, jeff::FloatUnaryOperation::_sinh>,
                     ConvertMathFloatUnaryOp<math::CoshOp, jeff::FloatUnaryOperation::_cosh>,
                     ConvertMathFloatUnaryOp<math::TanhOp, jeff::FloatUnaryOperation::_tanh>,
                     ConvertMathFloatUnaryOp<math::AsinhOp, jeff::FloatUnaryOperation::_asinh>,
                     ConvertMathFloatUnaryOp<math::AcoshOp, jeff::FloatUnaryOperation::_acosh>,
                     ConvertMathFloatUnaryOp<math::AtanhOp, jeff::FloatUnaryOperation::_atanh>,
                     ConvertMathFloatBinaryOp<math::Atan2Op, jeff::FloatBinaryOperation::_atan2>,
                     ConvertMathFloatBinaryOp<math::PowFOp, jeff::FloatBinaryOperation::_pow>,
                     ConvertMathFloatIsOp<math::IsNaNOp, jeff::FloatIsOperation::_isNan>,
                     ConvertMathFloatIsOp<math::IsInfOp, jeff::FloatIsOperation::_isInf>>(context);

        if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
            signalPassFailure();
            return;
        }
    }
};

} // namespace mlir
