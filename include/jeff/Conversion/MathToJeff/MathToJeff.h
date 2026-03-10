#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {
#define GEN_PASS_DECL_MATHTOJEFF
#include "jeff/Conversion/MathToJeff/MathToJeff.h.inc"

#define GEN_PASS_REGISTRATION
#include "jeff/Conversion/MathToJeff/MathToJeff.h.inc"
} // namespace mlir
