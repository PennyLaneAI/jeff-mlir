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

#include "jeff/IR/JeffDialect.h"

#include "jeff/IR/JeffOps.h"
#include "mlir/IR/DialectImplementation.h" // needed for generated type parser

#include "llvm/ADT/TypeSwitch.h" // needed for generated type parser

using namespace mlir;
using namespace mlir::jeff;

//===----------------------------------------------------------------------===//
// jeff dialect definitions.
//===----------------------------------------------------------------------===//

#include "jeff/IR/JeffOpsDialect.cpp.inc"

void JeffDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "jeff/IR/JeffOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "jeff/IR/JeffAttributes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "jeff/IR/JeffOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// jeff type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "jeff/IR/JeffOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "jeff/IR/JeffAttributes.cpp.inc"
