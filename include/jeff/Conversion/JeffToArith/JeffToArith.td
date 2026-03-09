include "mlir/Pass/PassBase.td"

def JeffToArith : Pass<"jeff-to-arith"> {
  let summary = "Convert Jeff operations to `arith` operations";

  let description = [{
    This pass converts all operations from the Jeff dialect to their equivalent operations in the `arith` dialect.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::jeff::JeffDialect",
  ];
}
