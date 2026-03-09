#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {
#define GEN_PASS_DECL_TENSORTOJEFF
#include "jeff/Conversion/TensorToJeff/TensorToJeff.h.inc"

#define GEN_PASS_REGISTRATION
#include "jeff/Conversion/TensorToJeff/TensorToJeff.h.inc"
} // namespace mlir
