include "mlir/Pass/PassBase.td"

def JeffToNative : Pass<"jeff-to-native"> {
  let summary = "Convert Jeff operations to built-in MLIR operations";

  let description = [{
    This pass converts all operations from the Jeff dialect to their equivalent operations in built-in MLIR dialects.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::jeff::JeffDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect",
  ];
}
