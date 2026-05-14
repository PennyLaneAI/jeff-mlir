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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>
#include <cstddef>

using namespace mlir;
using namespace mlir::jeff;

//===----------------------------------------------------------------------===//
// jeff op definitions.
//===----------------------------------------------------------------------===//

#include "jeff/IR/JeffEnums.cpp.inc"
#define GET_OP_CLASSES
#include "jeff/IR/JeffOps.cpp.inc"

namespace {

/**
 * @brief Converts an xor operation to a not operation if possible.
 */
struct XorToNot final : OpRewritePattern<IntBinaryOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(IntBinaryOp op, PatternRewriter& rewriter) const override {
        if (op.getOp() != IntBinaryOperation::_xor) {
            return failure();
        }
        auto* definingOp = op.getB().getDefiningOp();
        if (definingOp == nullptr) {
            return failure();
        }
        return llvm::TypeSwitch<Operation*, LogicalResult>(definingOp)
            .Case<IntConst1Op>([&](IntConst1Op b) {
                if (!b.getVal()) {
                    return failure();
                }
                rewriter.replaceOpWithNewOp<IntUnaryOp>(op, op.getA(), IntUnaryOperation::_not);
                if (b->use_empty()) {
                    rewriter.eraseOp(b);
                }
                return success();
            })
            .Case<IntConst8Op>([&](IntConst8Op b) {
                if (b.getVal() != 0xFF) {
                    return failure();
                }
                rewriter.replaceOpWithNewOp<IntUnaryOp>(op, op.getA(), IntUnaryOperation::_not);
                if (b->use_empty()) {
                    rewriter.eraseOp(b);
                }
                return success();
            })
            .Case<IntConst16Op>([&](IntConst16Op b) {
                if (b.getVal() != 0xFFFF) {
                    return failure();
                }
                rewriter.replaceOpWithNewOp<IntUnaryOp>(op, op.getA(), IntUnaryOperation::_not);
                if (b->use_empty()) {
                    rewriter.eraseOp(b);
                }
                return success();
            })
            .Case<IntConst32Op>([&](IntConst32Op b) {
                if (b.getVal() != 0xFFFFFFFF) {
                    return failure();
                }
                rewriter.replaceOpWithNewOp<IntUnaryOp>(op, op.getA(), IntUnaryOperation::_not);
                if (b->use_empty()) {
                    rewriter.eraseOp(b);
                }
                return success();
            })
            .Case<IntConst64Op>([&](IntConst64Op b) {
                if (b.getVal() != 0xFFFFFFFFFFFFFFFF) {
                    return failure();
                }
                rewriter.replaceOpWithNewOp<IntUnaryOp>(op, op.getA(), IntUnaryOperation::_not);
                if (b->use_empty()) {
                    rewriter.eraseOp(b);
                }
                return success();
            })
            .Default([&](auto) { return failure(); });
    }
};

// Adapted from
// https://github.com/llvm/llvm-project/blob/a58268a77cdbfeb0b71f3e76d169ddd7edf7a4df/mlir/lib/Dialect/SCF/IR/SCF.cpp#L480
void printInitializationList(OpAsmPrinter& p, Block::BlockArgListType blocksArgs,
                             ValueRange initializers, llvm::StringRef prefix = "") {
    assert(blocksArgs.size() == initializers.size() &&
           "expected same length of arguments and initializers");

    p << prefix << '(';
    llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p,
                          [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
    p << ")";
}

template <typename OpType>
LogicalResult verifyRegionArgs(OpType op, ValueRange inValues, ValueRange outValues,
                               Block::BlockArgListType regionArgs) {
    if (regionArgs.size() != outValues.size()) {
        return op.emitOpError("mismatch in number of basic block args and output values");
    }

    unsigned i = 0;
    for (auto e : llvm::zip(inValues, regionArgs, outValues)) {
        if (std::get<0>(e).getType() != std::get<2>(e).getType()) {
            return op.emitOpError()
                   << "types mismatch between " << i << "th iter operand and output value";
        }
        if (std::get<1>(e).getType() != std::get<2>(e).getType()) {
            return op.emitOpError()
                   << "types mismatch between " << i << "th iter region arg and output value";
        }

        ++i;
    }

    return success();
}

} // namespace

void IntBinaryOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
    results.add<XorToNot>(context);
}

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

ParseResult SwitchOp::parse(OpAsmParser& /*parser*/, OperationState& /*result*/) {
    // TODO: Implement this.
    // TODO: When implementing the parser, also relax the verifier — the
    //   current `in_values.size() == out_values.size()` requirement (enforced
    //   via `verifyRegionArgs`) is a carry-over from loop iter-args and is
    //   not part of the intended semantics for a switch.
    llvm::report_fatal_error("SwitchOp::parse is not implemented yet");
}

LogicalResult SwitchOp::verify() {
    if (getInValues().size() != getNumResults()) {
        return emitOpError("mismatch in number of input and output values");
    }

    return success();
}

LogicalResult SwitchOp::verifyRegions() {
    llvm::SmallVector<Region*> regions;
    auto branches = getBranches();
    regions.reserve(1 + branches.size());
    regions.push_back(&getDefault());
    for (auto& branch : branches) {
        regions.push_back(&branch);
    }

    auto inValues = getInValues();
    auto outValues = getOutValues();

    for (auto& region : regions) {
        auto regionArgs = region->getArguments();
        if (verifyRegionArgs(*this, inValues, outValues, regionArgs).failed()) {
            return failure();
        }
    }

    return success();
}

// Adapted from
// https://github.com/llvm/llvm-project/blob/a58268a77cdbfeb0b71f3e76d169ddd7edf7a4df/mlir/lib/Dialect/SCF/IR/SCF.cpp#L496
void ForOp::print(OpAsmPrinter& p) {
    auto inValues = getInValues();
    auto inductionVar = getBody().getArgument(0);
    auto regionArgs = getBody().getArguments().drop_front(1);

    p << " " << inductionVar << " = " << getStart() << " to " << getStop() << " step " << getStep();

    if (!inValues.empty()) {
        printInitializationList(p, regionArgs, inValues, " args");
        p << " -> (" << inValues.getTypes() << ')';
    }

    p << " : " << inductionVar.getType() << ' ';

    p.printRegion(getRegion(),
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/!inValues.empty());
    p.printOptionalAttrDict((*this)->getAttrs());
}

// Adapted from
// https://github.com/llvm/llvm-project/blob/a58268a77cdbfeb0b71f3e76d169ddd7edf7a4df/mlir/lib/Dialect/SCF/IR/SCF.cpp#L516
ParseResult ForOp::parse(OpAsmParser& parser, OperationState& result) {
    auto& builder = parser.getBuilder();
    Type type;

    OpAsmParser::Argument inductionVar;
    OpAsmParser::UnresolvedOperand start;
    OpAsmParser::UnresolvedOperand stop;
    OpAsmParser::UnresolvedOperand step;

    // Parse the induction variable followed by '='.
    if (parser.parseOperand(inductionVar.ssaName) || parser.parseEqual() ||
        // Parse loop bounds.
        parser.parseOperand(start) || parser.parseKeyword("to") || parser.parseOperand(stop) ||
        parser.parseKeyword("step") || parser.parseOperand(step)) {
        return failure();
    }

    // Parse the optional initial iteration arguments.
    llvm::SmallVector<OpAsmParser::Argument, 4> regionArgs;
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
    regionArgs.push_back(inductionVar);

    // Parse assignment list and result types list.
    bool hasArgs = succeeded(parser.parseOptionalKeyword("args"));
    if (hasArgs) {
        if (parser.parseAssignmentList(regionArgs, operands) ||
            parser.parseArrowTypeList(result.types)) {
            return failure();
        }
    }

    if (regionArgs.size() != result.types.size() + 1) {
        return parser.emitError(parser.getNameLoc(),
                                "mismatch in number of loop-carried values and defined values");
    }

    // Parse type.
    if (parser.parseColon() || parser.parseType(type)) {
        return failure();
    }

    // Set block argument types so that they are known when parsing the region.
    regionArgs.front().type = type;
    for (auto [arg, argType] : llvm::zip_equal(llvm::drop_begin(regionArgs), result.types)) {
        arg.type = argType;
    }

    // Parse the body region.
    Region* body = result.addRegion();
    if (parser.parseRegion(*body, regionArgs)) {
        return failure();
    }
    ForOp::ensureTerminator(*body, builder, result.location);

    // Resolve input operands.
    if (parser.resolveOperand(start, type, result.operands) ||
        parser.resolveOperand(stop, type, result.operands) ||
        parser.resolveOperand(step, type, result.operands)) {
        return failure();
    }
    if (hasArgs) {
        for (auto argOperandType :
             llvm::zip_equal(llvm::drop_begin(regionArgs), operands, result.types)) {
            Type argOpType = std::get<2>(argOperandType);
            std::get<0>(argOperandType).type = argOpType;
            if (parser.resolveOperand(std::get<1>(argOperandType), argOpType, result.operands)) {
                return failure();
            }
        }
    }

    // Parse the optional attribute list.
    if (parser.parseOptionalAttrDict(result.attributes)) {
        return failure();
    }

    return success();
}

// Adapted from
// https://github.com/llvm/llvm-project/blob/a58268a77cdbfeb0b71f3e76d169ddd7edf7a4df/mlir/lib/Dialect/SCF/IR/SCF.cpp#L350
LogicalResult ForOp::verify() {
    if (getInValues().size() != getNumResults()) {
        return emitOpError("mismatch in number of input and output values");
    }

    return success();
}

// Adapted from
// https://github.com/llvm/llvm-project/blob/a58268a77cdbfeb0b71f3e76d169ddd7edf7a4df/mlir/lib/Dialect/SCF/IR/SCF.cpp#L359
LogicalResult ForOp::verifyRegions() {
    auto inductionVar = getBody().getArgument(0);
    if (inductionVar.getType() != getStart().getType()) {
        return emitOpError("expected induction variable to be same type as bounds and step");
    }

    auto inValues = getInValues();
    auto outValues = getOutValues();
    auto regionArgs = getBody().getArguments().drop_front(1);
    if (verifyRegionArgs(*this, inValues, outValues, regionArgs).failed()) {
        return failure();
    }

    return success();
}

// Adapted from
// https://github.com/llvm/llvm-project/blob/a58268a77cdbfeb0b71f3e76d169ddd7edf7a4df/mlir/lib/Dialect/SCF/IR/SCF.cpp#L3343
void WhileOp::print(OpAsmPrinter& p) {
    auto inValues = getInValues();

    // Emit `: ( types )` only when there are in-values.
    if (!inValues.empty()) {
        p << " : (" << inValues.getTypes() << ")";
    }

    // Condition region: `args ( $assignments )` then the region.
    auto& condition = getCondition();
    auto conditionArgs = condition.getArguments();
    printInitializationList(p, conditionArgs, inValues, " args");
    p << ' ';
    p.printRegion(condition, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);

    // Body region: `args ( $assignments )` then the region.
    auto& body = getBody();
    auto bodyArgs = body.getArguments();
    printInitializationList(p, bodyArgs, inValues, " args");
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/!inValues.empty());

    p.printOptionalAttrDict((*this)->getAttrs());
}

// Adapted from
// https://github.com/llvm/llvm-project/blob/a58268a77cdbfeb0b71f3e76d169ddd7edf7a4df/mlir/lib/Dialect/SCF/IR/SCF.cpp#L3303
ParseResult WhileOp::parse(OpAsmParser& parser, OperationState& result) {
    auto& builder = parser.getBuilder();

    Region* condition = result.addRegion();
    Region* body = result.addRegion();

    // Parse optional `: ( types )`. Omitted when there are no in-values.
    llvm::SmallVector<Type, 4> types;
    if (succeeded(parser.parseOptionalColon())) {
        if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
                return parser.parseType(types.emplace_back());
            })) {
            return failure();
        }
    }

    // Parse the condition region's `args ( $assignments )`.
    llvm::SmallVector<OpAsmParser::Argument, 4> condRegionArgs;
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> condOperands;
    if (parser.parseKeyword("args") || parser.parseAssignmentList(condRegionArgs, condOperands)) {
        return failure();
    }

    if (condRegionArgs.size() != types.size()) {
        return parser.emitError(parser.getNameLoc())
               << "expected " << types.size() << " condition arguments but got "
               << condRegionArgs.size();
    }

    for (auto [arg, ty] : llvm::zip_equal(condRegionArgs, types)) {
        arg.type = ty;
    }

    if (parser.parseRegion(*condition, condRegionArgs)) {
        return failure();
    }
    WhileOp::ensureTerminator(*condition, builder, result.location);

    // Parse the body region's `args ( $assignments )`.
    llvm::SmallVector<OpAsmParser::Argument, 4> bodyRegionArgs;
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> bodyOperands;
    if (parser.parseKeyword("args") || parser.parseAssignmentList(bodyRegionArgs, bodyOperands)) {
        return failure();
    }

    if (bodyRegionArgs.size() != types.size()) {
        return parser.emitError(parser.getNameLoc())
               << "expected " << types.size() << " body arguments but got "
               << bodyRegionArgs.size();
    }

    // Both args clauses must reference the same operands.
    // Sizes are already equal at this point (both equal types.size()), so just compare names.
    for (auto [c, b] : llvm::zip_equal(condOperands, bodyOperands)) {
        if (c.name != b.name || c.number != b.number) {
            auto cOperand = llvm::formatv("%{0}{1}", c.name,
                (c.number > 0) ? llvm::formatv("#{0}", c.number).str() : "");
            auto bOperand = llvm::formatv("%{0}{1}", b.name,
                (b.number > 0) ? llvm::formatv("#{0}", b.number).str() : "");
            return parser.emitError(parser.getNameLoc())
                   << "condition and body args must bind the same operands "
                   << "(got " << bOperand << ", expected " << cOperand << ")";
        }
    }

    for (auto [arg, ty] : llvm::zip_equal(bodyRegionArgs, types)) {
        arg.type = ty;
    }

    if (parser.parseRegion(*body, bodyRegionArgs)) {
        return failure();
    }
    WhileOp::ensureTerminator(*body, builder, result.location);

    // Resolve operands (condition and body operand lists are equal).
    if (parser.resolveOperands(bodyOperands, types, parser.getCurrentLocation(), result.operands)) {
        return failure();
    }

    // Op results have the same types as in-values.
    result.addTypes(types);

    if (parser.parseOptionalAttrDict(result.attributes)) {
        return failure();
    }

    return success();
}

LogicalResult WhileOp::verify() {
    if (getInValues().size() != getNumResults()) {
        return emitOpError("mismatch in number of input and output values");
    }

    return success();
}

LogicalResult WhileOp::verifyRegions() {
    auto inValues = getInValues();
    auto outValues = getOutValues();

    auto conditionArgs = getCondition().getArguments();
    if (verifyRegionArgs(*this, inValues, outValues, conditionArgs).failed()) {
        return failure();
    }

    auto bodyArgs = getBody().getArguments();
    if (verifyRegionArgs(*this, inValues, outValues, bodyArgs).failed()) {
        return failure();
    }

    return success();
}

void DoWhileOp::print(OpAsmPrinter& p) {
    auto inValues = getInValues();

    // Emit `: ( types )` only when there are in-values.
    if (!inValues.empty()) {
        p << " : (" << inValues.getTypes() << ")";
    }

    // Body region: `args ( $assignments )` then the region.
    auto& body = getBody();
    auto bodyArgs = body.getArguments();
    printInitializationList(p, bodyArgs, inValues, " args");
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/!inValues.empty());

    // Condition region: `args ( $assignments )` then the region.
    auto& condition = getCondition();
    auto conditionArgs = condition.getArguments();
    printInitializationList(p, conditionArgs, inValues, " args");
    p << ' ';
    p.printRegion(condition, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);

    p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult DoWhileOp::parse(OpAsmParser& parser, OperationState& result) {
    auto& builder = parser.getBuilder();

    Region* body = result.addRegion();
    Region* condition = result.addRegion();

    // Parse optional `: ( types )`. Omitted when there are no in-values.
    llvm::SmallVector<Type, 4> types;
    if (succeeded(parser.parseOptionalColon())) {
        if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
                return parser.parseType(types.emplace_back());
            })) {
            return failure();
        }
    }

    // Parse the body region's `args ( $assignments )`.
    llvm::SmallVector<OpAsmParser::Argument, 4> bodyRegionArgs;
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> bodyOperands;
    if (parser.parseKeyword("args") || parser.parseAssignmentList(bodyRegionArgs, bodyOperands)) {
        return failure();
    }

    if (bodyRegionArgs.size() != types.size()) {
        return parser.emitError(parser.getNameLoc())
               << "expected " << types.size() << " body arguments but got "
               << bodyRegionArgs.size();
    }

    for (auto [arg, ty] : llvm::zip_equal(bodyRegionArgs, types)) {
        arg.type = ty;
    }

    if (parser.parseRegion(*body, bodyRegionArgs)) {
        return failure();
    }
    WhileOp::ensureTerminator(*body, builder, result.location);

    // Parse the condition region's `args ( $assignments )`.
    llvm::SmallVector<OpAsmParser::Argument, 4> condRegionArgs;
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> condOperands;
    if (parser.parseKeyword("args") || parser.parseAssignmentList(condRegionArgs, condOperands)) {
        return failure();
    }

    if (condRegionArgs.size() != types.size()) {
        return parser.emitError(parser.getNameLoc())
               << "expected " << types.size() << " condition arguments but got "
               << condRegionArgs.size();
    }

    // Both args clauses must reference the same operands.
    // Sizes are already equal at this point (both equal types.size()), so just compare names.
    for (auto [b, c] : llvm::zip_equal(bodyOperands, condOperands)) {
        if (b.name != c.name || b.number != c.number) {
            auto bOperand = llvm::formatv("%{0}{1}", b.name,
                (b.number > 0) ? llvm::formatv("#{0}", b.number).str() : "");
            auto cOperand = llvm::formatv("%{0}{1}", c.name,
                (c.number > 0) ? llvm::formatv("#{0}", c.number).str() : "");
            return parser.emitError(parser.getNameLoc())
                   << "body and condition args must bind the same operands "
                   << "(got " << cOperand << ", expected " << bOperand << ")";
        }
    }

    for (auto [arg, ty] : llvm::zip_equal(condRegionArgs, types)) {
        arg.type = ty;
    }

    if (parser.parseRegion(*condition, condRegionArgs)) {
        return failure();
    }
    WhileOp::ensureTerminator(*condition, builder, result.location);

    // Resolve operands (condition and body operand lists are equal).
    if (parser.resolveOperands(condOperands, types, parser.getCurrentLocation(), result.operands)) {
        return failure();
    }

    // Op results have the same types as in-values.
    result.addTypes(types);

    if (parser.parseOptionalAttrDict(result.attributes)) {
        return failure();
    }

    return success();
}

LogicalResult DoWhileOp::verify() {
    if (getInValues().size() != getNumResults()) {
        return emitOpError("mismatch in number of input and output values");
    }

    return success();
}

LogicalResult DoWhileOp::verifyRegions() {
    auto inValues = getInValues();
    auto outValues = getOutValues();

    auto conditionArgs = getCondition().getArguments();
    if (verifyRegionArgs(*this, inValues, outValues, conditionArgs).failed()) {
        return failure();
    }

    auto bodyArgs = getBody().getArguments();
    if (verifyRegionArgs(*this, inValues, outValues, bodyArgs).failed()) {
        return failure();
    }

    return success();
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
