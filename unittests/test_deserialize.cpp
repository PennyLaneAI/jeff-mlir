#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"

#include <algorithm>
#include <filesystem>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <vector>

namespace fs = std::filesystem;

int main() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::jeff::JeffDialect, mlir::func::FuncDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  const fs::path inputsDir = "unittests/inputs";

  std::vector<fs::path> jeffFiles;
  for (const auto& entry : fs::directory_iterator(inputsDir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto& path = entry.path();
    if (path.extension() == ".jeff") {
      jeffFiles.push_back(path);
    }
  }
  std::sort(jeffFiles.begin(), jeffFiles.end());

  for (const auto& path : jeffFiles) {
    llvm::outs() << "Deserializing " << path.filename().string() << "...\n";
    auto module = deserialize(&context, path.string());
    module->print(llvm::outs());
    llvm::outs() << "\n\n";
  }

  return 0;
}
