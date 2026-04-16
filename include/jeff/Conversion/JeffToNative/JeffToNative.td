include "mlir/Pass/PassBase.td"

def JeffToNative : Pass<"jeff-to-native"> {
  let summary = "Convert `jeff` operations to built-in MLIR operations";

  let description = [{
    This pass converts all operations from `jeff`'s `Int_Op`, `IntArray_Op`, `Float_Op`, and `FloatArray_Op` classes to their equivalent operations in built-in MLIR dialects.

    Known limitations:

    - `IntArrayConst*Op`s and `FloatArrayConst*Op`s with dynamic size are not supported.
    - `IntArrayCreateOp`s and `FloatArrayCreateOp`s with dynamic size are not supported.
    - Only one-dimensional tensors are supported.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect",
  ];
}
