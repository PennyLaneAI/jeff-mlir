#pragma once

#include <capnp/serialize.h>
#include <kj/array.h>
#include <mlir/IR/BuiltinOps.h>

/**
 * @brief Serialize an MLIR module into a .jeff file.
 * @param module The MLIR module to serialize.
 * @return A flat byte array containing the serialized module.
 */
kj::Array<capnp::word> serialize(mlir::ModuleOp module);
