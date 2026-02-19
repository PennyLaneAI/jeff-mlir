#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <string>

/**
 * @brief Deserialize a .jeff file into an MLIR module.
 * @param context The MLIR context to use for the deserialization.
 * @param path The path to the .jeff file.
 * @return An owning reference to the deserialized MLIR module.
 */
mlir::OwningOpRef<mlir::ModuleOp> deserialize(mlir::MLIRContext* context,
                                              const std::string& path);
