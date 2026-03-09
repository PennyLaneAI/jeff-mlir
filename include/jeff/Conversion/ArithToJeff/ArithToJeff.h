#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {
#define GEN_PASS_DECL_ARITHTOJEFF
#include "jeff/Conversion/ArithToJeff/ArithToJeff.h.inc"

#define GEN_PASS_REGISTRATION
#include "jeff/Conversion/ArithToJeff/ArithToJeff.h.inc"
} // namespace mlir
