#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {
#define GEN_PASS_DECL_JEFFTOARITH
#include "jeff/Conversion/JeffToArith/JeffToArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "jeff/Conversion/JeffToArith/JeffToArith.h.inc"
} // namespace mlir
