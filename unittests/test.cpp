#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>

int main() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::jeff::JeffDialect, mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect>();

  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::jeff::JeffDialect, mlir::func::FuncDialect,
                      mlir::tensor::TensorDialect>();

  llvm::outs() << "Deserializing entangled_calls.jeff...\n";
  auto module1 = deserialize(&context, "unittests/entangled_calls.jeff");
  module1->print(llvm::outs());
  llvm::outs() << "\n\n";

  llvm::outs() << "Deserializing entangled_qs.jeff...\n";
  auto module2 = deserialize(&context, "unittests/entangled_qs.jeff");
  module2->print(llvm::outs());
  llvm::outs() << "\n\n";

  llvm::outs() << "Deserializing python_optimization.jeff...\n";
  auto module3 = deserialize(&context, "unittests/python_optimization.jeff");
  module3->print(llvm::outs());

  return 0;
}
