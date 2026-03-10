#include "jeff/Conversion/TensorToJeff/TensorToJeff.h"

#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
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

#define GEN_PASS_DEF_TENSORTOJEFF
#include "jeff/Conversion/TensorToJeff/TensorToJeff.h.inc"

struct ConvertTensorEmptyOpToJeff final : OpConversionPattern<tensor::EmptyOp> {
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

struct ConvertTensorExtractOpToJeff final : OpConversionPattern<tensor::ExtractOp> {
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

struct ConvertTensorInsertOpToJeff final : OpConversionPattern<tensor::InsertOp> {
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

struct ConvertTensorDimOpToJeff final : OpConversionPattern<tensor::DimOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::DimOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        return llvm::TypeSwitch<Type, LogicalResult>(op.getSource().getType().getElementType())
            .Case<IntegerType>([&](auto) -> LogicalResult {
                rewriter.replaceOpWithNewOp<jeff::IntArrayLengthOp>(op, adaptor.getSource());
                return success();
            })
            .Case<FloatType>([&](auto) -> LogicalResult {
                rewriter.replaceOpWithNewOp<jeff::FloatArrayLengthOp>(op, adaptor.getSource());
                return success();
            })
            .Default([&](auto) -> LogicalResult {
                return rewriter.notifyMatchFailure(op, "Unsupported element type");
            });
    }
};

struct ConvertTensorFromElementsOpToJeff final : OpConversionPattern<tensor::FromElementsOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::FromElementsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        return llvm::TypeSwitch<Type, LogicalResult>(op.getType().getElementType())
            .Case<IntegerType>([&](auto) -> LogicalResult {
                rewriter.replaceOpWithNewOp<jeff::IntArrayCreateOp>(op, op.getType(),
                                                                    adaptor.getElements());
                return success();
            })
            .Case<FloatType>([&](auto) -> LogicalResult {
                rewriter.replaceOpWithNewOp<jeff::FloatArrayCreateOp>(op, op.getType(),
                                                                      adaptor.getElements());
                return success();
            })
            .Default([&](auto) -> LogicalResult {
                return rewriter.notifyMatchFailure(op, "Unsupported element type");
            });
    }
};

/**
 * @brief Pass for converting `tensor` operations to Jeff operations
 */
struct TensorToJeff final : impl::TensorToJeffBase<TensorToJeff> {
    using TensorToJeffBase::TensorToJeffBase;

  protected:
    void runOnOperation() override {
        MLIRContext* context = &getContext();
        auto* module = getOperation();

        ConversionTarget target(*context);
        target.addLegalDialect<jeff::JeffDialect>();
        target.addLegalOp<arith::ConstantOp, arith::IndexCastOp>();
        target.addIllegalDialect<tensor::TensorDialect>();

        RewritePatternSet patterns(context);
        patterns.add<ConvertTensorEmptyOpToJeff, ConvertTensorExtractOpToJeff,
                     ConvertTensorInsertOpToJeff, ConvertTensorDimOpToJeff,
                     ConvertTensorFromElementsOpToJeff>(context);

        if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
            signalPassFailure();
            return;
        }
    }
};

} // namespace mlir
