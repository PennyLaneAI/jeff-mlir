#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"

#include <mlir/IR/BuiltinOps.h>

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::jeff::JeffDialect>();

  auto module = deserialize(&context, "unittests/entangled_qs.jeff");
  module->dump();

  return 0;
}
