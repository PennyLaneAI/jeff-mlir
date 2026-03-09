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

template <typename ArithOp, jeff::IntBinaryOperation JeffOp>
struct ConvertArithIntBinaryOp final : OpConversionPattern<ArithOp> {
    using OpConversionPattern<ArithOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::IntBinaryOp>(op, adaptor.getLhs(), adaptor.getRhs(),
                                                       JeffOp);
        return success();
    }
};

template <typename ArithOp, jeff::FloatBinaryOperation JeffOp>
struct ConvertArithFloatBinaryOp final : OpConversionPattern<ArithOp> {
    using OpConversionPattern<ArithOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::FloatBinaryOp>(op, adaptor.getLhs(), adaptor.getRhs(),
                                                         JeffOp);
        return success();
    }
};

struct ConvertArithCmpIOpToJeff final : OpConversionPattern<arith::CmpIOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getLhs();
        auto b = adaptor.getRhs();
        switch (op.getPredicate()) {
        case arith::CmpIPredicate::eq:
            rewriter.replaceOpWithNewOp<jeff::IntComparisonOp>(op, a, b,
                                                               jeff::IntComparisonOperation::_eq);
            break;
        case arith::CmpIPredicate::slt:
            rewriter.replaceOpWithNewOp<jeff::IntComparisonOp>(op, a, b,
                                                               jeff::IntComparisonOperation::_ltS);
            break;
        case arith::CmpIPredicate::sle:
            rewriter.replaceOpWithNewOp<jeff::IntComparisonOp>(op, a, b,
                                                               jeff::IntComparisonOperation::_lteS);
            break;
        case arith::CmpIPredicate::ult:
            rewriter.replaceOpWithNewOp<jeff::IntComparisonOp>(op, a, b,
                                                               jeff::IntComparisonOperation::_ltU);
            break;
        case arith::CmpIPredicate::ule:
            rewriter.replaceOpWithNewOp<jeff::IntComparisonOp>(op, a, b,
                                                               jeff::IntComparisonOperation::_lteU);
            break;
        default:
            rewriter.notifyMatchFailure(op, "Unknown comparison operation");
        }
        return success();
    }
};

struct ConvertArithCmpFOpToJeff final : OpConversionPattern<arith::CmpFOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto a = adaptor.getLhs();
        auto b = adaptor.getRhs();
        switch (op.getPredicate()) {
        case arith::CmpFPredicate::OEQ:
            rewriter.replaceOpWithNewOp<jeff::FloatComparisonOp>(
                op, a, b, jeff::FloatComparisonOperation::_eq);
            break;
        case arith::CmpFPredicate::OLT:
            rewriter.replaceOpWithNewOp<jeff::FloatComparisonOp>(
                op, a, b, jeff::FloatComparisonOperation::_lt);
            break;
        case arith::CmpFPredicate::OLE:
            rewriter.replaceOpWithNewOp<jeff::FloatComparisonOp>(
                op, a, b, jeff::FloatComparisonOperation::_lte);
            break;
        default:
            rewriter.notifyMatchFailure(op, "Unknown comparison operation");
        }
        return success();
    }
};

struct ConvertArithConstOpToJeff final : OpConversionPattern<arith::ConstantOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
        auto value = op.getValue();
        return llvm::TypeSwitch<Type, LogicalResult>(op.getType())
            .Case<IntegerType>([&](auto type) -> LogicalResult {
                auto intAttr = llvm::dyn_cast<IntegerAttr>(value);
                if (!intAttr) {
                    return rewriter.notifyMatchFailure(op, "Expected IntegerAttr");
                }
                switch (type.getWidth()) {
                case 1:
                    rewriter.replaceOpWithNewOp<jeff::IntConst1Op>(op, intAttr);
                    return success();
                case 8:
                    rewriter.replaceOpWithNewOp<jeff::IntConst8Op>(op, intAttr);
                    return success();
                case 16:
                    rewriter.replaceOpWithNewOp<jeff::IntConst16Op>(op, intAttr);
                    return success();
                case 32:
                    rewriter.replaceOpWithNewOp<jeff::IntConst32Op>(op, intAttr);
                    return success();
                case 64:
                    rewriter.replaceOpWithNewOp<jeff::IntConst64Op>(op, intAttr);
                    return success();
                default:
                    return rewriter.notifyMatchFailure(op, "Unsupported integer type");
                }
            })
            .Case<FloatType>([&](auto type) -> LogicalResult {
                auto floatAttr = llvm::dyn_cast<FloatAttr>(value);
                if (!floatAttr) {
                    return rewriter.notifyMatchFailure(op, "Expected FloatAttr");
                }
                switch (type.getWidth()) {
                case 32:
                    rewriter.replaceOpWithNewOp<jeff::FloatConst32Op>(op, floatAttr);
                    return success();
                case 64:
                    rewriter.replaceOpWithNewOp<jeff::FloatConst64Op>(op, floatAttr);
                    return success();
                default:
                    return rewriter.notifyMatchFailure(op, "Unsupported float type");
                }
            })
            .Case<RankedTensorType>([&](auto type) -> LogicalResult {
                if (type.getRank() != 1) {
                    return rewriter.notifyMatchFailure(op, "Only 1D tensors are supported");
                }
                auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(value);
                if (!denseAttr) {
                    return rewriter.notifyMatchFailure(op, "Expected DenseElementsAttr");
                }
                auto elementType = type.getElementType();
                return llvm::TypeSwitch<Type, LogicalResult>(elementType)
                    .template Case<IntegerType>([&](auto intType) -> LogicalResult {
                        switch (intType.getWidth()) {
                        case 1: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<bool>());
                            auto inArrayAttr =
                                mlir::DenseBoolArrayAttr::get(op.getContext(), inArray);
                            rewriter.replaceOpWithNewOp<jeff::IntArrayConst1Op>(op, type,
                                                                                inArrayAttr);
                            return success();
                        }
                        case 8: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<int8_t>());
                            auto inArrayAttr =
                                mlir::DenseI8ArrayAttr::get(op.getContext(), inArray);
                            rewriter.replaceOpWithNewOp<jeff::IntArrayConst8Op>(op, type,
                                                                                inArrayAttr);
                            return success();
                        }
                        case 16: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<int16_t>());
                            auto inArrayAttr =
                                mlir::DenseI16ArrayAttr::get(op.getContext(), inArray);
                            rewriter.replaceOpWithNewOp<jeff::IntArrayConst16Op>(op, type,
                                                                                 inArrayAttr);
                            return success();
                        }
                        case 32: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<int32_t>());
                            auto inArrayAttr =
                                mlir::DenseI32ArrayAttr::get(op.getContext(), inArray);
                            rewriter.replaceOpWithNewOp<jeff::IntArrayConst32Op>(op, type,
                                                                                 inArrayAttr);
                            return success();
                        }
                        case 64: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<int64_t>());
                            auto inArrayAttr =
                                mlir::DenseI64ArrayAttr::get(op.getContext(), inArray);
                            rewriter.replaceOpWithNewOp<jeff::IntArrayConst64Op>(op, type,
                                                                                 inArrayAttr);
                            return success();
                        }
                        default:
                            return rewriter.notifyMatchFailure(op, "Unsupported integer type");
                        }
                    })
                    .template Case<FloatType>([&](auto floatType) -> LogicalResult {
                        switch (floatType.getWidth()) {
                        case 32: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<float>());
                            auto inArrayAttr =
                                mlir::DenseF32ArrayAttr::get(op.getContext(), inArray);
                            rewriter.replaceOpWithNewOp<jeff::FloatArrayConst32Op>(op, type,
                                                                                   inArrayAttr);
                            return success();
                        }
                        case 64: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<double>());
                            auto inArrayAttr =
                                mlir::DenseF64ArrayAttr::get(op.getContext(), inArray);
                            rewriter.replaceOpWithNewOp<jeff::FloatArrayConst64Op>(op, type,
                                                                                   inArrayAttr);
                            return success();
                        }
                        default:
                            return rewriter.notifyMatchFailure(op, "Unsupported float type");
                        }
                    })
                    .Default([&](auto) -> LogicalResult {
                        return rewriter.notifyMatchFailure(op, "Unsupported element type");
                    });
            })
            .Default([&](auto) -> LogicalResult {
                return rewriter.notifyMatchFailure(op, "Unsupported type");
            });
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
        target.addLegalDialect<jeff::JeffDialect>();

        target.addIllegalOp<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
                            arith::DivUIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
                            arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp,
                            arith::RemSIOp, arith::RemUIOp, arith::ShLIOp, arith::ShRSIOp,
                            arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::MaxNumFOp,
                            arith::MinNumFOp, arith::CmpIOp, arith::CmpFOp>();
        {
            RewritePatternSet patterns(context);
            patterns
                .add<ConvertArithIntBinaryOp<arith::AddIOp, jeff::IntBinaryOperation::_add>,
                     ConvertArithIntBinaryOp<arith::SubIOp, jeff::IntBinaryOperation::_sub>,
                     ConvertArithIntBinaryOp<arith::MulIOp, jeff::IntBinaryOperation::_mul>,
                     ConvertArithIntBinaryOp<arith::DivSIOp, jeff::IntBinaryOperation::_divS>,
                     ConvertArithIntBinaryOp<arith::DivUIOp, jeff::IntBinaryOperation::_divU>,
                     ConvertArithIntBinaryOp<arith::AndIOp, jeff::IntBinaryOperation::_and>,
                     ConvertArithIntBinaryOp<arith::OrIOp, jeff::IntBinaryOperation::_or>,
                     ConvertArithIntBinaryOp<arith::XOrIOp, jeff::IntBinaryOperation::_xor>,
                     ConvertArithIntBinaryOp<arith::MinSIOp, jeff::IntBinaryOperation::_minS>,
                     ConvertArithIntBinaryOp<arith::MinUIOp, jeff::IntBinaryOperation::_minU>,
                     ConvertArithIntBinaryOp<arith::MaxSIOp, jeff::IntBinaryOperation::_maxS>,
                     ConvertArithIntBinaryOp<arith::MaxUIOp, jeff::IntBinaryOperation::_maxU>,
                     ConvertArithIntBinaryOp<arith::RemSIOp, jeff::IntBinaryOperation::_remS>,
                     ConvertArithIntBinaryOp<arith::RemUIOp, jeff::IntBinaryOperation::_remU>,
                     ConvertArithIntBinaryOp<arith::ShLIOp, jeff::IntBinaryOperation::_shl>,
                     ConvertArithIntBinaryOp<arith::ShRSIOp, jeff::IntBinaryOperation::_shr>,
                     ConvertArithFloatBinaryOp<arith::AddFOp, jeff::FloatBinaryOperation::_add>,
                     ConvertArithFloatBinaryOp<arith::SubFOp, jeff::FloatBinaryOperation::_sub>,
                     ConvertArithFloatBinaryOp<arith::MulFOp, jeff::FloatBinaryOperation::_mul>,
                     ConvertArithFloatBinaryOp<arith::MaxNumFOp, jeff::FloatBinaryOperation::_max>,
                     ConvertArithFloatBinaryOp<arith::MinNumFOp, jeff::FloatBinaryOperation::_min>,
                     ConvertArithCmpIOpToJeff, ConvertArithCmpFOpToJeff>(context);
            if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
                signalPassFailure();
                return;
            }
        }

        target.addIllegalOp<arith::ConstantOp>();
        {
            RewritePatternSet patterns(context);
            patterns.add<ConvertArithConstOpToJeff>(context);
            if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
                signalPassFailure();
            }
        }
    }
};

} // namespace mlir
