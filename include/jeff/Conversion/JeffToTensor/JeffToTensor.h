#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {
#define GEN_PASS_DECL_JEFFTOTENSOR
#include "jeff/Conversion/JeffToTensor/JeffToTensor.h.inc"

#define GEN_PASS_REGISTRATION
#include "jeff/Conversion/JeffToTensor/JeffToTensor.h.inc"
} // namespace mlir
