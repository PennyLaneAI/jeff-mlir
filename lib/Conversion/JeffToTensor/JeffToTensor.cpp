#include "jeff/Conversion/JeffToTensor/JeffToTensor.h"

#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffOps.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
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

#define GEN_PASS_DEF_JEFFTOTENSOR
#include "jeff/Conversion/JeffToTensor/JeffToTensor.h.inc"

template <typename JeffOp> struct ConvertJeffArrayZeroOp final : OpConversionPattern<JeffOp> {
    using OpConversionPattern<JeffOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(JeffOp op, typename JeffOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto length = adaptor.getLength();
        auto lengthIndex =
            arith::IndexCastOp::create(rewriter, op.getLoc(), rewriter.getIndexType(), length);
        rewriter.replaceOpWithNewOp<tensor::EmptyOp>(op, op.getType(), lengthIndex.getResult());
        return success();
    }
};

template <typename JeffOp> struct ConvertJeffArrayGetIndexOp final : OpConversionPattern<JeffOp> {
    using OpConversionPattern<JeffOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(JeffOp op, typename JeffOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto index = adaptor.getIndex();
        auto indexIndex =
            arith::IndexCastOp::create(rewriter, op.getLoc(), rewriter.getIndexType(), index);
        rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, adaptor.getInArray(),
                                                       indexIndex.getResult());
        return success();
    }
};

template <typename JeffOp> struct ConvertJeffArraySetIndexOp final : OpConversionPattern<JeffOp> {
    using OpConversionPattern<JeffOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(JeffOp op, typename JeffOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto index = adaptor.getIndex();
        auto indexIndex =
            arith::IndexCastOp::create(rewriter, op.getLoc(), rewriter.getIndexType(), index);
        rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, adaptor.getValue(), adaptor.getInArray(),
                                                      indexIndex.getResult());
        return success();
    }
};

template <typename JeffOp> struct ConvertJeffArrayLengthOp final : OpConversionPattern<JeffOp> {
    using OpConversionPattern<JeffOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(JeffOp op, typename JeffOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto zero = arith::ConstantOp::create(rewriter, op.getLoc(), rewriter.getIndexAttr(0));
        rewriter.replaceOpWithNewOp<tensor::DimOp>(op, adaptor.getInArray(), zero.getResult());
        return success();
    }
};

template <typename JeffOp> struct ConvertJeffArrayCreateOp final : OpConversionPattern<JeffOp> {
    using OpConversionPattern<JeffOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(JeffOp op, typename JeffOp::Adaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, adaptor.getInArray());
        return success();
    }
};

/**
 * @brief Pass for converting Jeff operations to `tensor` operations
 */
struct JeffToTensor final : impl::JeffToTensorBase<JeffToTensor> {
    using JeffToTensorBase::JeffToTensorBase;

  protected:
    void runOnOperation() override {
        MLIRContext* context = &getContext();
        auto* module = getOperation();

        ConversionTarget target(*context);
        target.addLegalDialect<tensor::TensorDialect>();
        target.addLegalOp<arith::ConstantOp, arith::IndexCastOp>();
        target.addIllegalOp<jeff::IntArrayZeroOp, jeff::FloatArrayZeroOp, jeff::IntArrayGetIndexOp,
                            jeff::FloatArrayGetIndexOp, jeff::IntArraySetIndexOp,
                            jeff::FloatArraySetIndexOp, jeff::IntArrayLengthOp,
                            jeff::FloatArrayLengthOp, jeff::IntArrayCreateOp,
                            jeff::FloatArrayCreateOp>();

        RewritePatternSet patterns(context);
        patterns.add<ConvertJeffArrayZeroOp<jeff::IntArrayZeroOp>,
                     ConvertJeffArrayZeroOp<jeff::FloatArrayZeroOp>,
                     ConvertJeffArrayGetIndexOp<jeff::IntArrayGetIndexOp>,
                     ConvertJeffArrayGetIndexOp<jeff::FloatArrayGetIndexOp>,
                     ConvertJeffArraySetIndexOp<jeff::IntArraySetIndexOp>,
                     ConvertJeffArraySetIndexOp<jeff::FloatArraySetIndexOp>,
                     ConvertJeffArrayLengthOp<jeff::IntArrayLengthOp>,
                     ConvertJeffArrayLengthOp<jeff::FloatArrayLengthOp>,
                     ConvertJeffArrayCreateOp<jeff::IntArrayCreateOp>,
                     ConvertJeffArrayCreateOp<jeff::FloatArrayCreateOp>>(context);

        if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
            signalPassFailure();
        }
    }
};

} // namespace mlir
