#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {
#define GEN_PASS_DECL_NATIVETOJEFF
#include "jeff/Conversion/NativeToJeff/NativeToJeff.h.inc"

#define GEN_PASS_REGISTRATION
#include "jeff/Conversion/NativeToJeff/NativeToJeff.h.inc"
} // namespace mlir
