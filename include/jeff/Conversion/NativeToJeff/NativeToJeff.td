include "mlir/Pass/PassBase.td"

def NativeToJeff : Pass<"native-to-jeff"> {
  let summary = "Convert built-in MLIR operations to Jeff operations";

  let description = [{
    This pass converts all operations from built-in MLIR dialects to their equivalent operations in the Jeff dialect.

    Known limitations:

    - `arith.constant` operations of type `index` are converted to `jeff.int_const_32` operations.
    - `arith.index_cast` and `tensor.cast` operations are removed during conversion.
      This assumes that functions have a Jeff-compatible signature (e.g., a function must not return an `index`).
      The conversion ensures that the created operations are compatible with each other.
  }];

  let dependentDialects = [
    "mlir::jeff::JeffDialect",
  ];
}
