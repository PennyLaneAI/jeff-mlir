#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>

/**
 * @brief Serialize an MLIR module into a byte buffer.
 * @param module The MLIR module to serialize.
 * @return A byte buffer containing the serialized module.
 *
 * @details
 * Known limitations:
 *
 * - Only one-dimensional tensors with dynamic size are supported.
 */
llvm::SmallVector<uint8_t> serialize(mlir::ModuleOp module);

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
void serializeToFile(mlir::ModuleOp module, llvm::StringRef path);
