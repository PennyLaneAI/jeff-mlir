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

#include <cassert>
#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::jeff;

//===----------------------------------------------------------------------===//
// jeff op definitions.
//===----------------------------------------------------------------------===//

#include "jeff/IR/JeffEnums.cpp.inc"
#define GET_OP_CLASSES
#include "jeff/IR/JeffOps.cpp.inc"

namespace {

// Adapted from
// https://github.com/llvm/llvm-project/blob/a58268a77cdbfeb0b71f3e76d169ddd7edf7a4df/mlir/lib/Dialect/SCF/IR/SCF.cpp#L480
void printInitializationList(OpAsmPrinter& p,
                             Block::BlockArgListType blocksArgs,
                             ValueRange initializers,
                             llvm::StringRef prefix = "") {
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");

  p << prefix << '(';
  llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
    p << std::get<0>(it) << " = " << std::get<1>(it);
  });
  p << ")";
}

} // namespace

void SwitchOp::print(OpAsmPrinter& p) {
  auto inValues = getInValues();

  p << '(' << getSelection() << ")";
  p << " : " << getSelection().getType();
  p << " -> (" << inValues.getTypes() << ") ";

  auto branches = getBranches();
  for (size_t i = 0; i < branches.size(); ++i) {
    p.printNewline();
    p << "case " << i << ' ';
    auto& branch = branches[i];
    auto regionArgs = branch.getArguments();
    printInitializationList(p, regionArgs, inValues, "args");
    p << ' ';
    p.printRegion(branch, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/!inValues.empty());
  }

  auto& defaultRegion = getDefault();
  if (!defaultRegion.empty()) {
    p.printNewline();
    p << "default ";
    auto regionArgs = defaultRegion.getArguments();
    printInitializationList(p, regionArgs, inValues, "args");
    p << ' ';
    p.printRegion(defaultRegion, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/!inValues.empty());
  }

  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult SwitchOp::parse(OpAsmParser& /*parser*/,
                            OperationState& /*result*/) {
  // TODO: Implement this
  llvm::report_fatal_error("SwitchOp::parse is not implemented yet");
}

// Adapted from
// https://github.com/llvm/llvm-project/blob/a58268a77cdbfeb0b71f3e76d169ddd7edf7a4df/mlir/lib/Dialect/SCF/IR/SCF.cpp#L496
void ForOp::print(OpAsmPrinter& p) {
  auto inValues = getInValues();
  auto inductionVar = getBody().getArgument(0);
  auto regionArgs = getBody().getArguments().drop_front(1);

  p << " " << inductionVar << " = " << getStart() << " to " << getStop()
    << " step " << getStep();

  if (!inValues.empty()) {
    printInitializationList(p, regionArgs, inValues, " args");
    p << " -> (" << inValues.getTypes() << ')';
  }

  if (Type t = inductionVar.getType(); !t.isIndex()) {
    p << " : " << t << ' ';
  } else {
    p << ' ';
  }

  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!inValues.empty());
  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult ForOp::parse(OpAsmParser& /*parser*/, OperationState& /*result*/) {
  // TODO: Implement this
  llvm::report_fatal_error("ForOp::parse is not implemented yet");
}

void WhileOp::print(OpAsmPrinter& p) {
  auto inValues = getInValues();

  auto& condition = getCondition();
  auto conditionArgs = condition.getArguments();
  printInitializationList(p, conditionArgs, inValues, " args");
  p << " -> (" << IntegerType::get(getContext(), 1) << ") ";
  p.printRegion(condition, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!inValues.empty());

  auto& body = getBody();
  auto bodyArgs = body.getArguments();
  printInitializationList(p, bodyArgs, inValues, " args");
  p << " -> (" << inValues.getTypes() << ") ";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!inValues.empty());

  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult WhileOp::parse(OpAsmParser& /*parser*/,
                           OperationState& /*result*/) {
  // TODO: Implement this
  llvm::report_fatal_error("WhileOp::parse is not implemented yet");
}

void DoWhileOp::print(OpAsmPrinter& p) {
  auto inValues = getInValues();

  auto& body = getBody();
  auto bodyArgs = body.getArguments();
  printInitializationList(p, bodyArgs, inValues, " args");
  p << " -> (" << inValues.getTypes() << ") ";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!inValues.empty());

  auto& condition = getCondition();
  auto conditionArgs = condition.getArguments();
  printInitializationList(p, conditionArgs, inValues, " args");
  p << " -> (" << IntegerType::get(getContext(), 1) << ") ";
  p.printRegion(condition, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!inValues.empty());

  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult DoWhileOp::parse(OpAsmParser& /*parser*/,
                             OperationState& /*result*/) {
  // TODO: Implement this
  llvm::report_fatal_error("DoWhileOp::parse is not implemented yet");
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
