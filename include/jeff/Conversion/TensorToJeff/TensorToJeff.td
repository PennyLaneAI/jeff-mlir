include "mlir/Pass/PassBase.td"

def TensorToJeff : Pass<"tensor-to-jeff"> {
  let summary = "Convert `tensor` operations to Jeff operations";

  let description = [{
    This pass converts all operations from the `tensor` dialect to their equivalent operations in the Jeff dialect.
  }];

  let dependentDialects = [
    "mlir::jeff::JeffDialect",
    "mlir::tensor::TensorDialect",
  ];
}
