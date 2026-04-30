#pragma once

#include <capnp/common.h>
#include <kj/array.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>

/**
 * @brief Serialize an MLIR module into a flat word array.
 * @param module The MLIR module to serialize.
 * @return A flat word array containing the serialized module.
 *
 * @details
 * Known limitations:
 *
 * - Only one-dimensional tensors with dynamic size are supported.
 */
kj::Array<capnp::word> serialize(mlir::ModuleOp module);

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
