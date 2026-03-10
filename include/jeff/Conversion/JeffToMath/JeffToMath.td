include "mlir/Pass/PassBase.td"

def JeffToMath : Pass<"jeff-to-math"> {
  let summary = "Convert Jeff operations to `math` operations";

  let description = [{
    This pass converts all operations from the Jeff dialect to their equivalent operations in the `math` dialect.
  }];

  let dependentDialects = [
    "mlir::jeff::JeffDialect",
    "mlir::math::MathDialect",
  ];
}
