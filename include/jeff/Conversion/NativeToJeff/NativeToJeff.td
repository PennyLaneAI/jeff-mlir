include "mlir/Pass/PassBase.td"

def NativeToJeff : Pass<"native-to-jeff"> {
  let summary = "Convert built-in MLIR operations to Jeff operations";

  let description = [{
    This pass converts all operations from built-in MLIR dialects to their equivalent operations in the Jeff dialect.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::jeff::JeffDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect",
  ];
}
