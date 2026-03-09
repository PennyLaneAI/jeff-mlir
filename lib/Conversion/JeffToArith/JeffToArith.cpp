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

/**
 * @brief Converts jeff.int_const1 to arith.constant
 *
 * @par Example:
 * ```mlir
 * %0 = jeff.int_const1(true) : i1
 * ```
 * is converted to
 * ```mlir
 * %0 = arith.constant true : i1
 * ```
 */
struct ConvertJeffIntConst1OpToArith final : OpConversionPattern<jeff::IntConst1Op> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::IntConst1Op op, OpAdaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValAttr());
        return success();
    }
};

/**
 * @brief Converts jeff.int_const64 to arith.constant
 *
 * @par Example:
 * ```mlir
 * %0 = jeff.int_const64(3) : i64
 * ```
 * is converted to
 * ```mlir
 * %0 = arith.constant 3 : i64
 * ```
 */
struct ConvertJeffIntConst64OpToArith final : OpConversionPattern<jeff::IntConst64Op> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::IntConst64Op op, OpAdaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValAttr());
        return success();
    }
};

/**
 * @brief Converts jeff.float_const64 to arith.constant
 *
 * @par Example:
 * ```mlir
 * %0 = jeff.float_const64(0.3) : f64
 * ```
 * is converted to
 * ```mlir
 * %0 = arith.constant 0.3 : f64
 * ```
 */
struct ConvertJeffFloatConst64OpToArith final : OpConversionPattern<jeff::FloatConst64Op> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(jeff::FloatConst64Op op, OpAdaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValAttr());
        return success();
    }
};

/**
 * @brief Type converter for Jeff-to-`arith` conversion
 */
class JeffToArithTypeConverter final : public TypeConverter {
  public:
    explicit JeffToArithTypeConverter(MLIRContext* ctx) {
        // Identity conversion for all types by default
        addConversion([](Type type) { return type; });
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
        RewritePatternSet patterns(context);
        const JeffToArithTypeConverter typeConverter(context);

        // Configure conversion target
        target.addIllegalDialect<jeff::JeffDialect>();
        target.addLegalDialect<arith::ArithDialect>();

        // Register operation conversion patterns
        patterns.add<ConvertJeffIntConst1OpToArith, ConvertJeffIntConst64OpToArith,
                     ConvertJeffFloatConst64OpToArith>(typeConverter, context);

        // Apply the conversion
        if (applyPartialConversion(module, target, std::move(patterns)).failed()) {
            signalPassFailure();
        }
    }
};

} // namespace mlir
