#pragma once

#include <llvm/Support/MemoryBuffer.h>
#include <mlir/IR/BuiltinOps.h>

#include <memory>

/**
 * @brief Serialize an MLIR module containing a jeff program into a memory buffer.
 * @param module The MLIR module to serialize.
 * @return An owned memory buffer containing the serialized jeff module.
 *
 * @details
 * Known limitations:
 *
 * - Only one-dimensional tensors with dynamic size are supported.
 */
std::unique_ptr<llvm::MemoryBuffer> serialize(mlir::ModuleOp module);

/**
 * @brief Serialize an MLIR module containing a jeff program into a .jeff file.
 * @param module The MLIR module to serialize.
 * @param path The path to the .jeff file.
 *
 * @details
 * Known limitations:
 *
 * - Only one-dimensional tensors with dynamic size are supported.
 */
void serializeToFile(mlir::ModuleOp module, llvm::StringRef path);
