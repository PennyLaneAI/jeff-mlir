include "mlir/Pass/PassBase.td"

def JeffToNative : Pass<"jeff-to-native"> {
  let summary = "Convert Jeff operations to built-in MLIR operations";

  let description = [{
    This pass converts all operations from the Jeff dialect to their equivalent operations in built-in MLIR dialects.

    Known limitations:

    - Only one-dimensional tensors are supported.
    - `tensor.cast` operations may be introduced during conversion to ensure a valid IR.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect",
  ];
}
