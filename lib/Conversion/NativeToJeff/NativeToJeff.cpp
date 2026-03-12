#include "jeff/Conversion/NativeToJeff/NativeToJeff.h"

#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
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

#define GEN_PASS_DEF_NATIVETOJEFF
#include "jeff/Conversion/NativeToJeff/NativeToJeff.h.inc"

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

struct ConvertArithConstOp final : OpConversionPattern<arith::ConstantOp> {
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
                auto jeffType =
                    mlir::RankedTensorType::get({mlir::ShapedType::kDynamic}, elementType);
                auto* ctx = op.getContext();
                auto loc = op.getLoc();
                return llvm::TypeSwitch<Type, LogicalResult>(elementType)
                    .template Case<IntegerType>([&](auto intType) -> LogicalResult {
                        switch (intType.getWidth()) {
                        case 1: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<bool>());
                            auto inArrayAttr = mlir::DenseBoolArrayAttr::get(ctx, inArray);
                            auto array = jeff::IntArrayConst1Op::create(rewriter, loc, jeffType,
                                                                        inArrayAttr);
                            rewriter.replaceOpWithNewOp<tensor::CastOp>(op, type,
                                                                        array.getOutArray());
                            return success();
                        }
                        case 8: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<int8_t>());
                            auto inArrayAttr = mlir::DenseI8ArrayAttr::get(ctx, inArray);
                            auto array = jeff::IntArrayConst8Op::create(rewriter, loc, jeffType,
                                                                        inArrayAttr);
                            rewriter.replaceOpWithNewOp<tensor::CastOp>(op, type,
                                                                        array.getOutArray());
                            return success();
                        }
                        case 16: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<int16_t>());
                            auto inArrayAttr = mlir::DenseI16ArrayAttr::get(ctx, inArray);
                            auto array = jeff::IntArrayConst16Op::create(rewriter, loc, jeffType,
                                                                         inArrayAttr);
                            rewriter.replaceOpWithNewOp<tensor::CastOp>(op, type,
                                                                        array.getOutArray());
                            return success();
                        }
                        case 32: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<int32_t>());
                            auto inArrayAttr = mlir::DenseI32ArrayAttr::get(ctx, inArray);
                            auto array = jeff::IntArrayConst32Op::create(rewriter, loc, jeffType,
                                                                         inArrayAttr);
                            rewriter.replaceOpWithNewOp<tensor::CastOp>(op, type,
                                                                        array.getOutArray());
                            return success();
                        }
                        case 64: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<int64_t>());
                            auto inArrayAttr = mlir::DenseI64ArrayAttr::get(ctx, inArray);
                            auto array = jeff::IntArrayConst64Op::create(rewriter, loc, jeffType,
                                                                         inArrayAttr);
                            rewriter.replaceOpWithNewOp<tensor::CastOp>(op, type,
                                                                        array.getOutArray());
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
                            auto inArrayAttr = mlir::DenseF32ArrayAttr::get(ctx, inArray);
                            auto array = jeff::FloatArrayConst32Op::create(rewriter, loc, jeffType,
                                                                           inArrayAttr);
                            rewriter.replaceOpWithNewOp<tensor::CastOp>(op, type,
                                                                        array.getOutArray());
                            return success();
                        }
                        case 64: {
                            auto inArray = llvm::to_vector(denseAttr.getValues<double>());
                            auto inArrayAttr = mlir::DenseF64ArrayAttr::get(ctx, inArray);
                            auto array = jeff::FloatArrayConst64Op::create(rewriter, loc, jeffType,
                                                                           inArrayAttr);
                            rewriter.replaceOpWithNewOp<tensor::CastOp>(op, type,
                                                                        array.getOutArray());
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
            .Default([&](auto) -> LogicalResult { return success(); });
    }
};

//===----------------------------------------------------------------------===//
// Int operations
//===----------------------------------------------------------------------===//

struct ConvertMathAbsIOp final : OpConversionPattern<math::AbsIOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(math::AbsIOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::IntUnaryOp>(op, adaptor.getOperand(),
                                                      jeff::IntUnaryOperation::_abs);
        return success();
    }
};

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

struct ConvertMathIPowIOp final : OpConversionPattern<math::IPowIOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(math::IPowIOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::IntBinaryOp>(op, adaptor.getLhs(), adaptor.getRhs(),
                                                       jeff::IntBinaryOperation::_pow);
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

//===----------------------------------------------------------------------===//
// Float operations
//===----------------------------------------------------------------------===//

template <typename MathOp, jeff::FloatUnaryOperation JeffOp>
struct ConvertMathFloatUnaryOp final : OpConversionPattern<MathOp> {
    using OpConversionPattern<MathOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(MathOp op, typename MathOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::FloatUnaryOp>(op, adaptor.getOperand(), JeffOp);
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

struct ConvertArithCmpFOp final : OpConversionPattern<arith::CmpFOp> {
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

template <typename MathOp, jeff::FloatIsOperation JeffOp>
struct ConvertMathFloatIsOp final : OpConversionPattern<MathOp> {
    using OpConversionPattern<MathOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(MathOp op, typename MathOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<jeff::FloatIsOp>(op, adaptor.getOperand(), JeffOp);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// IntArray/FloatArray operations
//===----------------------------------------------------------------------===//

struct ConvertTensorEmptyOp final : OpConversionPattern<tensor::EmptyOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto sizes = adaptor.getDynamicSizes();
        if (sizes.size() != 1) {
            return rewriter.notifyMatchFailure(op, "Only 1D tensors are supported");
        }
        auto sizeInt =
            arith::IndexCastOp::create(rewriter, op.getLoc(), rewriter.getI32Type(), sizes[0]);
        return llvm::TypeSwitch<Type, LogicalResult>(op.getType().getElementType())
            .Case<IntegerType>([&](auto) -> LogicalResult {
                rewriter.replaceOpWithNewOp<jeff::IntArrayZeroOp>(op, op.getType(),
                                                                  sizeInt.getResult());
                return success();
            })
            .Case<FloatType>([&](auto) -> LogicalResult {
                rewriter.replaceOpWithNewOp<jeff::FloatArrayZeroOp>(op, op.getType(),
                                                                    sizeInt.getResult());
                return success();
            })
            .Default([&](auto) -> LogicalResult {
                return rewriter.notifyMatchFailure(op, "Unsupported element type");
            });
    }
};

struct ConvertTensorExtractOp final : OpConversionPattern<tensor::ExtractOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::ExtractOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto indices = adaptor.getIndices();
        if (indices.size() != 1) {
            return rewriter.notifyMatchFailure(op, "Only 1D tensors are supported");
        }
        auto indexInt =
            arith::IndexCastOp::create(rewriter, op.getLoc(), rewriter.getI32Type(), indices[0]);
        return llvm::TypeSwitch<Type, LogicalResult>(op.getType())
            .Case<IntegerType>([&](auto) -> LogicalResult {
                rewriter.replaceOpWithNewOp<jeff::IntArrayGetIndexOp>(
                    op, op.getType(), adaptor.getTensor(), indexInt.getResult());
                return success();
            })
            .Case<FloatType>([&](auto) -> LogicalResult {
                rewriter.replaceOpWithNewOp<jeff::FloatArrayGetIndexOp>(
                    op, op.getType(), adaptor.getTensor(), indexInt.getResult());
                return success();
            })
            .Default([&](auto) -> LogicalResult {
                return rewriter.notifyMatchFailure(op, "Unsupported element type");
            });
    }
};

struct ConvertTensorInsertOp final : OpConversionPattern<tensor::InsertOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::InsertOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto indices = adaptor.getIndices();
        if (indices.size() != 1) {
            return rewriter.notifyMatchFailure(op, "Only 1D tensors are supported");
        }
        auto indexInt =
            arith::IndexCastOp::create(rewriter, op.getLoc(), rewriter.getI32Type(), indices[0]);
        return llvm::TypeSwitch<Type, LogicalResult>(op.getType().getElementType())
            .Case<IntegerType>([&](auto) -> LogicalResult {
                rewriter.replaceOpWithNewOp<jeff::IntArraySetIndexOp>(
                    op, op.getType(), adaptor.getDest(), indexInt.getResult(), adaptor.getScalar());
                return success();
            })
            .Case<FloatType>([&](auto) -> LogicalResult {
                rewriter.replaceOpWithNewOp<jeff::FloatArraySetIndexOp>(
                    op, op.getType(), adaptor.getDest(), indexInt.getResult(), adaptor.getScalar());
                return success();
            })
            .Default([&](auto) -> LogicalResult {
                return rewriter.notifyMatchFailure(op, "Unsupported element type");
            });
    }
};

struct ConvertTensorDimOp final : OpConversionPattern<tensor::DimOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::DimOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        // TODO: Add check
        rewriter.eraseOp(op.getIndex().getDefiningOp());
        return llvm::TypeSwitch<Type, LogicalResult>(op.getSource().getType().getElementType())
            .Case<IntegerType>([&](auto) -> LogicalResult {
                auto length =
                    jeff::IntArrayLengthOp::create(rewriter, op.getLoc(), adaptor.getSource());
                rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, op.getType(),
                                                                length.getResult());
                return success();
            })
            .Case<FloatType>([&](auto) -> LogicalResult {
                auto length =
                    jeff::FloatArrayLengthOp::create(rewriter, op.getLoc(), adaptor.getSource());
                rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, op.getType(),
                                                                length.getResult());
                return success();
            })
            .Default([&](auto) -> LogicalResult {
                return rewriter.notifyMatchFailure(op, "Unsupported element type");
            });
    }
};

struct ConvertTensorFromElementsOp final : OpConversionPattern<tensor::FromElementsOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::FromElementsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto type = op.getType();
        auto elementType = type.getElementType();
        auto jeffType = mlir::RankedTensorType::get({mlir::ShapedType::kDynamic}, elementType);
        return llvm::TypeSwitch<Type, LogicalResult>(elementType)
            .Case<IntegerType>([&](auto) -> LogicalResult {
                auto array = jeff::IntArrayCreateOp::create(rewriter, op.getLoc(), jeffType,
                                                            adaptor.getElements());
                rewriter.replaceOpWithNewOp<tensor::CastOp>(op, type, array.getOutArray());
                return success();
            })
            .Case<FloatType>([&](auto) -> LogicalResult {
                auto array = jeff::FloatArrayCreateOp::create(rewriter, op.getLoc(), jeffType,
                                                              adaptor.getElements());
                rewriter.replaceOpWithNewOp<tensor::CastOp>(op, type, array.getOutArray());
                return success();
            })
            .Default([&](auto) -> LogicalResult {
                return rewriter.notifyMatchFailure(op, "Unsupported element type");
            });
    }
};

struct ConvertTensorCastOp final : OpConversionPattern<tensor::CastOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::CastOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto predecessor = op.getSource().getDefiningOp<tensor::CastOp>();
        if (!predecessor) {
            return success();
        }
        if (predecessor.getSource().getType() != op.getType()) {
            return success();
        }
        rewriter.replaceOp(op, predecessor.getSource());
        rewriter.eraseOp(predecessor);
        return success();
    }
};

/**
 * @brief Pass for converting built-in MLIR operations to Jeff operations
 */
struct NativeToJeff final : impl::NativeToJeffBase<NativeToJeff> {
    using NativeToJeffBase::NativeToJeffBase;

  protected:
    void runOnOperation() override {
        MLIRContext* context = &getContext();
        auto* module = getOperation();

        ConversionTarget target(*context);
        target.addIllegalDialect<arith::ArithDialect, math::MathDialect, tensor::TensorDialect>();
        target.addLegalOp<arith::IndexCastOp, tensor::CastOp>();
        target.addLegalDialect<jeff::JeffDialect>();

        RewritePatternSet patterns(context);
        patterns.add<
            // Constants
            ConvertArithConstOp,
            // Int operations
            ConvertMathAbsIOp,
            ConvertArithIntBinaryOp<arith::AddIOp, jeff::IntBinaryOperation::_add>,
            ConvertArithIntBinaryOp<arith::SubIOp, jeff::IntBinaryOperation::_sub>,
            ConvertArithIntBinaryOp<arith::MulIOp, jeff::IntBinaryOperation::_mul>,
            ConvertArithIntBinaryOp<arith::DivSIOp, jeff::IntBinaryOperation::_divS>,
            ConvertArithIntBinaryOp<arith::DivUIOp, jeff::IntBinaryOperation::_divU>,
            ConvertMathIPowIOp,
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
            ConvertArithCmpIOpToJeff,
            // Float operations
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
            ConvertArithFloatBinaryOp<arith::AddFOp, jeff::FloatBinaryOperation::_add>,
            ConvertArithFloatBinaryOp<arith::SubFOp, jeff::FloatBinaryOperation::_sub>,
            ConvertArithFloatBinaryOp<arith::MulFOp, jeff::FloatBinaryOperation::_mul>,
            ConvertMathFloatBinaryOp<math::Atan2Op, jeff::FloatBinaryOperation::_atan2>,
            ConvertMathFloatBinaryOp<math::PowFOp, jeff::FloatBinaryOperation::_pow>,
            ConvertArithFloatBinaryOp<arith::MaxNumFOp, jeff::FloatBinaryOperation::_max>,
            ConvertArithFloatBinaryOp<arith::MinNumFOp, jeff::FloatBinaryOperation::_min>,
            ConvertArithCmpFOp, ConvertMathFloatIsOp<math::IsNaNOp, jeff::FloatIsOperation::_isNan>,
            ConvertMathFloatIsOp<math::IsInfOp, jeff::FloatIsOperation::_isInf>,
            // IntArray/FloatArray operations
            ConvertTensorEmptyOp, ConvertTensorExtractOp, ConvertTensorInsertOp, ConvertTensorDimOp,
            ConvertTensorFromElementsOp>(context);

        if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
            signalPassFailure();
            return;
        }

        // Try to remove tensor::CastOp introduced during conversion of tensor::FromElementsOp
        target.addIllegalOp<tensor::CastOp>();
        {
            RewritePatternSet patterns(context);
            patterns.add<ConvertTensorCastOp>(context);
            if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
                signalPassFailure();
            }
        }
    }
};

} // namespace mlir
