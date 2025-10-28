// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffOps.h"

using namespace mlir;
using namespace jeff;

//===----------------------------------------------------------------------===//
// jeff op definitions.
//===----------------------------------------------------------------------===//

#include "jeff/IR/JeffEnums.cpp.inc"
#define GET_OP_CLASSES
#include "jeff/IR/JeffOps.cpp.inc"

//===----------------------------------------------------------------------===//
// jeff op builders.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// jeff op canonicalizers.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// jeff op verifiers.
//===----------------------------------------------------------------------===//
