#pragma once

#include <capnp/common.h>
#include <kj/array.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

/**
 * @brief Deserialize a flat word array into an MLIR module.
 * @param context The MLIR context to use for the deserialization.
 * @param data A flat word array containing the serialized module.
 * @return An owning reference to the deserialized MLIR module.
 */
mlir::OwningOpRef<mlir::ModuleOp> deserialize(mlir::MLIRContext* context,
                                              kj::ArrayPtr<capnp::word> data);

/**
 * @brief Deserialize a .jeff file into an MLIR module.
 * @param context The MLIR context to use for the deserialization.
 * @param path The path to the .jeff file.
 * @return An owning reference to the deserialized MLIR module.
 */
mlir::OwningOpRef<mlir::ModuleOp> deserializeFromFile(mlir::MLIRContext* context,
                                                      llvm::StringRef path);
