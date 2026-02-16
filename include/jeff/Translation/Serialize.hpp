#pragma once

#include <capnp/serialize.h>
#include <kj/array.h>
#include <mlir/IR/BuiltinOps.h>

kj::Array<capnp::word> serialize(mlir::ModuleOp module);
