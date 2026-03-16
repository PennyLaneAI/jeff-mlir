#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {
#define GEN_PASS_DECL_JEFFTONATIVE
#include "jeff/Conversion/JeffToNative/JeffToNative.h.inc"

#define GEN_PASS_REGISTRATION
#include "jeff/Conversion/JeffToNative/JeffToNative.h.inc"
} // namespace mlir
