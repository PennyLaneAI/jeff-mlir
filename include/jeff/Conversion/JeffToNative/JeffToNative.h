#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {

class RewritePatternSet;

#define GEN_PASS_DECL_JEFFTONATIVE
#include "jeff/Conversion/JeffToNative/JeffToNative.h.inc"

#define GEN_PASS_REGISTRATION
#include "jeff/Conversion/JeffToNative/JeffToNative.h.inc"

namespace jeff {

void populateJeffToNativeConversionPatterns(RewritePatternSet& patterns);

} // namespace jeff

} // namespace mlir
