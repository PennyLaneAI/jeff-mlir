#pragma once

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>

/**
 * @brief Serialize an MLIR module into a .jeff file.
 * @param module The MLIR module to serialize.
 * @param path The path to the .jeff file.
 *
 * @details
 * Known limitations:
 *
 * - Only one-dimensional tensors with dynamic size are supported.
 */
void serialize(mlir::ModuleOp module, llvm::StringRef path);
