#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {

class RewritePatternSet;

#define GEN_PASS_DECL_NATIVETOJEFF
#include "jeff/Conversion/NativeToJeff/NativeToJeff.h.inc"

#define GEN_PASS_REGISTRATION
#include "jeff/Conversion/NativeToJeff/NativeToJeff.h.inc"

namespace jeff {

void populateNativeToJeffConversionPatterns(RewritePatternSet& patterns);

} // namespace jeff

} // namespace mlir
