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

#include "jeff/IR/JeffOps.h"

#include "jeff/IR/JeffDialect.h"

#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <optional>

using namespace mlir;
using namespace mlir::jeff;

//===----------------------------------------------------------------------===//
// jeff op definitions.
//===----------------------------------------------------------------------===//

#include "jeff/IR/JeffEnums.cpp.inc"
#define GET_OP_CLASSES
#include "jeff/IR/JeffOps.cpp.inc"

namespace {

void printInitializationList(OpAsmPrinter& p,
                             Block::BlockArgListType blocksArgs,
                             ValueRange initializers, StringRef prefix = "") {
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");
  if (initializers.empty())
    return;

  p << prefix << '(';
  llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
    p << std::get<0>(it) << " = " << std::get<1>(it);
  });
  p << ")";
}

} // namespace

// Adapted from
// https://github.com/llvm/llvm-project/blob/a58268a77cdbfeb0b71f3e76d169ddd7edf7a4df/mlir/lib/Dialect/SCF/IR/SCF.cpp#L496
void ForOp::print(OpAsmPrinter& p) {
  auto inductionVar = getBody().getArgument(0);
  auto regionIterArgs = getBody().getArguments().drop_front(1);

  p << " " << inductionVar << " = " << getStart() << " to " << getStop()
    << " step " << getStep();

  printInitializationList(p, regionIterArgs, getInValues(), " args");
  if (!getInValues().empty()) {
    p << " -> (" << getInValues().getTypes() << ')';
  }

  if (Type t = inductionVar.getType(); !t.isIndex()) {
    p << " : " << t << ' ';
  } else {
    p << ' ';
  }

  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!getInValues().empty());
  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult ForOp::parse(OpAsmParser& parser, OperationState& result) {
  llvm::report_fatal_error("ForOp::parse is not implemented yet");
}

//===----------------------------------------------------------------------===//
// jeff op builders.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// jeff op canonicalizers.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// jeff op verifiers.
//===----------------------------------------------------------------------===//
