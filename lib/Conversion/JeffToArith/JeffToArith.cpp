#include "jeff/Conversion/JeffToArith/JeffToArith.h"

#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffOps.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
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

#define GEN_PASS_DEF_JEFFTOARITH
#include "jeff/Conversion/JeffToArith/JeffToArith.h.inc"

template <typename ConstOp> struct ConvertJeffIntConstOp final : OpConversionPattern<ConstOp> {
    using OpConversionPattern<ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ConstOp op, typename ConstOp::Adaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValAttr());
        return success();
    }
};

template <typename ConstOp> struct ConvertJeffIntArrayConstOp final : OpConversionPattern<ConstOp> {
    using OpConversionPattern<ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ConstOp op, typename ConstOp::Adaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
        auto denseAttr = DenseElementsAttr::get(op.getType(), op.getInArrayAttr().asArrayRef());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, denseAttr);
        return success();
    }
};

template <typename ConstOp> struct ConvertJeffFloatConstOp final : OpConversionPattern<ConstOp> {
    using OpConversionPattern<ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ConstOp op, typename ConstOp::Adaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValAttr());
        return success();
    }
};

template <typename ConstOp>
struct ConvertJeffFloatArrayConstOp final : OpConversionPattern<ConstOp> {
    using OpConversionPattern<ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ConstOp op, typename ConstOp::Adaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
        auto denseAttr = DenseElementsAttr::get(op.getType(), op.getInArrayAttr().asArrayRef());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, denseAttr);
        return success();
    }
};

struct ConvertJeffIntBinaryOpToArith final : OpConversionPattern<jeff::IntBinaryOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::IntBinaryOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        auto b = adaptor.getB();
        switch (op.getOp()) {
        case jeff::IntBinaryOperation::_add:
            rewriter.replaceOpWithNewOp<arith::AddIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_sub:
            rewriter.replaceOpWithNewOp<arith::SubIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_mul:
            rewriter.replaceOpWithNewOp<arith::MulIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_divS:
            rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_divU:
            rewriter.replaceOpWithNewOp<arith::DivUIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_and:
            rewriter.replaceOpWithNewOp<arith::AndIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_or:
            rewriter.replaceOpWithNewOp<arith::OrIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_xor:
            rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_minS:
            rewriter.replaceOpWithNewOp<arith::MinSIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_minU:
            rewriter.replaceOpWithNewOp<arith::MinUIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_maxS:
            rewriter.replaceOpWithNewOp<arith::MaxSIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_maxU:
            rewriter.replaceOpWithNewOp<arith::MaxUIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_remS:
            rewriter.replaceOpWithNewOp<arith::RemSIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_remU:
            rewriter.replaceOpWithNewOp<arith::RemUIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_shl:
            rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, a, b);
            break;
        case jeff::IntBinaryOperation::_shr:
            rewriter.replaceOpWithNewOp<arith::ShRSIOp>(op, a, b);
            break;
        default:
            rewriter.notifyMatchFailure(op, "Unknown binary operation");
        }
        return success();
    }
};

struct ConvertJeffIntComparisonOpToArith final : OpConversionPattern<jeff::IntComparisonOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::IntComparisonOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        auto b = adaptor.getB();
        switch (op.getOp()) {
        case jeff::IntComparisonOperation::_eq:
            rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::eq, a, b);
            break;
        case jeff::IntComparisonOperation::_ltS:
            rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::slt, a, b);
            break;
        case jeff::IntComparisonOperation::_lteS:
            rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::sle, a, b);
            break;
        case jeff::IntComparisonOperation::_ltU:
            rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::ult, a, b);
            break;
        case jeff::IntComparisonOperation::_lteU:
            rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::ule, a, b);
            break;
        default:
            rewriter.notifyMatchFailure(op, "Unknown comparison operation");
        }
        return success();
    }
};

struct ConvertJeffFloatBinaryOpToArith final : OpConversionPattern<jeff::FloatBinaryOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::FloatBinaryOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        auto b = adaptor.getB();
        switch (op.getOp()) {
        case jeff::FloatBinaryOperation::_add:
            rewriter.replaceOpWithNewOp<arith::AddFOp>(op, a, b);
            break;
        case jeff::FloatBinaryOperation::_sub:
            rewriter.replaceOpWithNewOp<arith::SubFOp>(op, a, b);
            break;
        case jeff::FloatBinaryOperation::_mul:
            rewriter.replaceOpWithNewOp<arith::MulFOp>(op, a, b);
            break;
        case jeff::FloatBinaryOperation::_max:
            rewriter.replaceOpWithNewOp<arith::MaxNumFOp>(op, a, b);
            break;
        case jeff::FloatBinaryOperation::_min:
            rewriter.replaceOpWithNewOp<arith::MinNumFOp>(op, a, b);
            break;
        default:
            rewriter.notifyMatchFailure(op, "Unknown binary operation");
        }
        return success();
    }
};

struct ConvertJeffFloatComparisonOpToArith final : OpConversionPattern<jeff::FloatComparisonOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::FloatComparisonOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getA();
        auto b = adaptor.getB();
        switch (op.getOp()) {
        case jeff::FloatComparisonOperation::_eq:
            rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, arith::CmpFPredicate::OEQ, a, b);
            break;
        case jeff::FloatComparisonOperation::_lt:
            rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, arith::CmpFPredicate::OLT, a, b);
            break;
        case jeff::FloatComparisonOperation::_lte:
            rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, arith::CmpFPredicate::OLE, a, b);
            break;
        default:
            rewriter.notifyMatchFailure(op, "Unknown comparison operation");
        }
        return success();
    }
};

/**
 * @brief Pass for converting Jeff operations to `arith` operations
 */
struct JeffToArith final : impl::JeffToArithBase<JeffToArith> {
    using JeffToArithBase::JeffToArithBase;

  protected:
    void runOnOperation() override {
        MLIRContext* context = &getContext();
        auto* module = getOperation();

        ConversionTarget target(*context);
        target.addIllegalOp<jeff::IntConst1Op, jeff::IntConst8Op, jeff::IntConst16Op,
                            jeff::IntConst32Op, jeff::IntConst64Op, jeff::IntComparisonOp,
                            jeff::FloatConst32Op, jeff::FloatConst64Op, jeff::FloatComparisonOp>();
        target.addDynamicallyLegalOp<jeff::IntBinaryOp>([](auto* op) {
            auto intOp = llvm::cast<jeff::IntBinaryOp>(op);
            auto binaryOp = intOp.getOp();
            if (binaryOp == jeff::IntBinaryOperation::_pow) {
                return true;
            }
            return false;
        });
        target.addDynamicallyLegalOp<jeff::FloatBinaryOp>([](auto* op) {
            auto floatOp = llvm::cast<jeff::FloatBinaryOp>(op);
            auto binaryOp = floatOp.getOp();
            if (binaryOp == jeff::FloatBinaryOperation::_pow ||
                binaryOp == jeff::FloatBinaryOperation::_atan2) {
                return true;
            }
            return false;
        });
        target.addLegalDialect<arith::ArithDialect>();

        RewritePatternSet patterns(context);
        patterns.add<
            ConvertJeffIntConstOp<jeff::IntConst1Op>, ConvertJeffIntConstOp<jeff::IntConst8Op>,
            ConvertJeffIntConstOp<jeff::IntConst16Op>, ConvertJeffIntConstOp<jeff::IntConst32Op>,
            ConvertJeffIntConstOp<jeff::IntConst64Op>,
            ConvertJeffIntArrayConstOp<jeff::IntArrayConst1Op>,
            ConvertJeffIntArrayConstOp<jeff::IntArrayConst8Op>,
            ConvertJeffIntArrayConstOp<jeff::IntArrayConst16Op>,
            ConvertJeffIntArrayConstOp<jeff::IntArrayConst32Op>,
            ConvertJeffIntArrayConstOp<jeff::IntArrayConst64Op>,
            ConvertJeffFloatConstOp<jeff::FloatConst32Op>,
            ConvertJeffFloatConstOp<jeff::FloatConst64Op>,
            ConvertJeffFloatArrayConstOp<jeff::FloatArrayConst32Op>,
            ConvertJeffFloatArrayConstOp<jeff::FloatArrayConst64Op>, ConvertJeffIntBinaryOpToArith,
            ConvertJeffIntComparisonOpToArith, ConvertJeffFloatBinaryOpToArith,
            ConvertJeffFloatComparisonOpToArith>(context);

        if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
            signalPassFailure();
        }
    }
};

} // namespace mlir
