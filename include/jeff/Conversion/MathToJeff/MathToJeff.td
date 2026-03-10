include "mlir/Pass/PassBase.td"

def MathToJeff : Pass<"math-to-jeff"> {
  let summary = "Convert `math` operations to Jeff operations";

  let description = [{
    This pass converts all operations from the `math` dialect to their equivalent operations in the Jeff dialect.
  }];

  let dependentDialects = [
    "mlir::jeff::JeffDialect",
    "mlir::math::MathDialect",
  ];
}
