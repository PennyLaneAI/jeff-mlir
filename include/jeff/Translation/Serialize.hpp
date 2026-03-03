#pragma once

#include <capnp/serialize.h>
#include <kj/array.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>

/**
 * @brief Serialize an MLIR module into a `kj::Array`.
 * @param module The MLIR module to serialize.
 * @return A `kj::Array` containing the serialized module.
 */
kj::Array<capnp::word> serializeToArray(mlir::ModuleOp module);

/**
 * @brief Serialize an MLIR module into a .jeff file.
 * @param module The MLIR module to serialize.
 * @param path The path to the .jeff file.
 */
void serialize(mlir::ModuleOp module, llvm::StringRef path);
