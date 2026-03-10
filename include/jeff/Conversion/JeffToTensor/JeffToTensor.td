include "mlir/Pass/PassBase.td"

def JeffToTensor : Pass<"jeff-to-tensor"> {
  let summary = "Convert Jeff operations to `tensor` operations";

  let description = [{
    This pass converts all operations from the Jeff dialect to their equivalent operations in the `tensor` dialect.
  }];

  let dependentDialects = [
    "mlir::jeff::JeffDialect",
    "mlir::tensor::TensorDialect",
  ];
}
