#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {
#define GEN_PASS_DECL_JEFFTOMATH
#include "jeff/Conversion/JeffToMath/JeffToMath.h.inc"

#define GEN_PASS_REGISTRATION
#include "jeff/Conversion/JeffToMath/JeffToMath.h.inc"
} // namespace mlir
