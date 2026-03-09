include "mlir/Pass/PassBase.td"

def ArithToJeff : Pass<"arith-to-jeff"> {
  let summary = "Convert `arith` operations to Jeff operations";

  let description = [{
    This pass converts all operations from the `arith` dialect to their equivalent operations in the Jeff dialect.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::jeff::JeffDialect",
  ];
}
