#include "jeff/IR/JeffDialect.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(int argc, char** argv) {
  mlir::registerAllPasses();
  // TODO: Register jeff passes here.

  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<mlir::jeff::JeffDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "jeff-mlir optimizer driver\n", registry));
}
