#pragma once

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>

/**
 * @brief Serialize an MLIR module into a .jeff file.
 * @param module The MLIR module to serialize.
 * @param path The path to the .jeff file.
 */
void serialize(mlir::ModuleOp module, llvm::StringRef path);
