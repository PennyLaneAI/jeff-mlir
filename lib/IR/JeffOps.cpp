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
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>
#include <cstdint>

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
    // The op's operand list is `selection` followed by `in_values`, in that
    // order, so we can stream both together.
    auto operands = (*this)->getOperands();
    auto resultTypes = getResultTypes();

    // Header: `(%sel, %a, %b) : (i32, T_in...) -> (T_out...)`.
    p << " (";
    llvm::interleaveComma(operands, p);
    p << ") : (";
    llvm::interleaveComma(operands.getTypes(), p);
    p << ") -> (" << resultTypes << ")";

    auto printRegionWithArgs = [&](Region& region) {
        p << " args(";
        llvm::interleaveComma(region.getArguments(), p);
        p << ") ";
        p.printRegion(region, /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/!resultTypes.empty());
    };

    for (auto [i, branch] : llvm::enumerate(getBranches())) {
        p.printNewline();
        p << "case " << i;
        printRegionWithArgs(branch);
    }

    auto& defaultRegion = getDefault();
    if (!defaultRegion.empty()) {
        p.printNewline();
        p << "default";
        printRegionWithArgs(defaultRegion);
    }

    p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult SwitchOp::parse(OpAsmParser& parser, OperationState& result) {
    auto& builder = parser.getBuilder();

    // The op declares `$default` first and `$branches` after,
    // so the default region must occupy index 0.
    // Pre-allocate it; populate later if present.
    Region* defaultRegion = result.addRegion();

    // Parse `(%sel, %a, %b)` — selector first, then in-values.
    llvm::SmallVector<OpAsmParser::UnresolvedOperand> operands;
    if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
            return parser.parseOperand(operands.emplace_back());
        })) {
        return failure();
    }
    if (operands.empty()) {
        return parser.emitError(parser.getNameLoc(), "expected at least the selector operand");
    }

    // Parse `: (T_sel, T_in...)` — operand types, in the same order.
    llvm::SmallVector<Type> operandTypes;
    if (parser.parseColon() || parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
            return parser.parseType(operandTypes.emplace_back());
        })) {
        return failure();
    }
    if (operandTypes.size() != operands.size()) {
        return parser.emitError(parser.getNameLoc())
               << "expected " << operands.size() << " operand types but got "
               << operandTypes.size();
    }

    // Parse `-> (T_out...)` — result types, independent of operand types.
    llvm::SmallVector<Type> resultTypes;
    if (parser.parseArrow() || parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
            return parser.parseType(resultTypes.emplace_back());
        })) {
        return failure();
    }

    if (parser.resolveOperands(operands, operandTypes, parser.getCurrentLocation(),
                               result.operands)) {
        return failure();
    }

    // In-value types are everything after the selector.
    auto inValueTypes = llvm::ArrayRef<Type>(operandTypes).drop_front(1);

    // Helper that parses `args(%x, %y) { ... }` into a region.
    auto parseRegionWithArgs = [&](Region& region) -> ParseResult {
        llvm::SmallVector<OpAsmParser::Argument> regionArgs;
        if (parser.parseKeyword("args") ||
            parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
                return parser.parseArgument(regionArgs.emplace_back());
            })) {
            return failure();
        }
        if (regionArgs.size() != inValueTypes.size()) {
            return parser.emitError(parser.getNameLoc())
                   << "expected " << inValueTypes.size() << " region arguments but got "
                   << regionArgs.size();
        }
        for (auto [arg, ty] : llvm::zip_equal(regionArgs, inValueTypes)) {
            arg.type = ty;
        }
        if (parser.parseRegion(region, regionArgs)) {
            return failure();
        }
        SwitchOp::ensureTerminator(region, builder, result.location);
        return success();
    };

    // Parse `case N args(...) { ... }` while the keyword is present.
    // Case labels are positional; require them to be 0, 1, 2, ...
    // so that print-then-parse round-trips faithfully.
    int64_t expectedCase = 0;
    while (succeeded(parser.parseOptionalKeyword("case"))) {
        int64_t caseValue = 0;
        if (parser.parseInteger(caseValue)) {
            return failure();
        }
        if (caseValue != expectedCase) {
            return parser.emitError(parser.getNameLoc())
                   << "expected `case " << expectedCase << "` but got `case " << caseValue << "`";
        }
        ++expectedCase;

        Region* branch = result.addRegion();
        if (parseRegionWithArgs(*branch)) {
            return failure();
        }
    }

    // Optional `default args(...) { ... }`.
    if (succeeded(parser.parseOptionalKeyword("default"))) {
        if (parseRegionWithArgs(*defaultRegion)) {
            return failure();
        }
    }

    result.addTypes(resultTypes);

    if (parser.parseOptionalAttrDict(result.attributes)) {
        return failure();
    }

    return success();
}

/**
 * @brief Verifies the 'case' and 'default' regions of a `jeff.switch`.
 *
 * `in_values` and `out_values` are independent for switch: there is no count or type relationship
 * between them. Each region's block arguments mirror `in_values`, and each region's `jeff.yield`
 * mirrors the op's results - two separate checks.
 */
LogicalResult SwitchOp::verifyRegions() {
    auto inValueTypes = getInValues().getTypes();
    auto resultTypes = getResultTypes();

    // Helper that verifies one region (a `case` body or the `default`).
    auto verifyRegion = [&](Region& region, const llvm::Twine& name) -> LogicalResult {
        // The parser always pre-allocates the default region,
        // leaving it empty when the source has no `default { ... }` clause.
        if (region.empty()) {
            return success();
        }

        // Block-argument count matches the `in_values` count.
        auto regionArgs = region.getArguments();
        if (regionArgs.size() != inValueTypes.size()) {
            return emitOpError() << name << " region has " << regionArgs.size()
                                 << " block arguments but op has " << inValueTypes.size()
                                 << " in-values";
        }
        // Block-argument types match the `in_values` types.
        for (auto [i, regionArg, inTy] : llvm::enumerate(regionArgs, inValueTypes)) {
            if (regionArg.getType() != inTy) {
                return emitOpError() << name << " region block argument " << i
                                     << " type does not match the corresponding in-value type";
            }
        }

        // `jeff.yield` is present.
        auto yield = dyn_cast<YieldOp>(region.front().back());
        if (!yield) {
            return emitOpError() << name << " region must terminate with `jeff.yield`";
        }
        // `jeff.yield` operand count matches the op's result count.
        if (yield.getNumOperands() != resultTypes.size()) {
            return emitOpError() << name << " region yields " << yield.getNumOperands()
                                 << " values but op has " << resultTypes.size() << " results";
        }
        // `jeff.yield` operand types match the op's result types.
        for (auto [i, yieldOp, resTy] : llvm::enumerate(yield.getOperands(), resultTypes)) {
            if (yieldOp.getType() != resTy) {
                return emitOpError() << name << " region yield operand " << i
                                     << " type does not match the corresponding result type";
            }
        }
        return success();
    };

    // Verify `case` regions.
    for (auto [i, branch] : llvm::enumerate(getBranches())) {
        if (verifyRegion(branch, "case " + llvm::Twine(i)).failed()) {
            return failure();
        }
    }
    // Verify `default` region.
    if (verifyRegion(getDefault(), "default").failed()) {
        return failure();
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
    llvm::SmallVector<OpAsmParser::Argument> regionArgs;
    llvm::SmallVector<OpAsmParser::UnresolvedOperand> operands;
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
    auto inductionVarType = inductionVar.getType();
    if (inductionVarType != getStart().getType()) {
        return emitOpError("expected induction variable to be same type as start");
    }
    if (inductionVarType != getStop().getType()) {
        return emitOpError("expected induction variable to be same type as stop");
    }
    if (inductionVarType != getStep().getType()) {
        return emitOpError("expected induction variable to be same type as step");
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

    // before region: `args ( $assignments )`.
    // Define the mapping between operands and block arguments.
    auto& before = getBefore();
    auto beforeArgs = before.getArguments();
    printInitializationList(p, beforeArgs, inValues, " args");
    p << ' ';
    p.printRegion(before, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);

    // after region: `args ( $names )`.
    // Block arguments only. The operands are already stated in the before region's `args(...)`.
    auto& after = getAfter();
    auto afterArgs = after.getArguments();
    p << " args(";
    llvm::interleaveComma(afterArgs, p);
    p << ") ";
    p.printRegion(after, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/!inValues.empty());

    p.printOptionalAttrDict((*this)->getAttrs());
}

// Adapted from
// https://github.com/llvm/llvm-project/blob/a58268a77cdbfeb0b71f3e76d169ddd7edf7a4df/mlir/lib/Dialect/SCF/IR/SCF.cpp#L3303
ParseResult WhileOp::parse(OpAsmParser& parser, OperationState& result) {
    auto& builder = parser.getBuilder();

    auto* before = result.addRegion();
    auto* after = result.addRegion();

    // Parse optional `: ( types )`.
    // Omitted when there are no in-values.
    llvm::SmallVector<Type> types;
    if (succeeded(parser.parseOptionalColon())) {
        if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
                return parser.parseType(types.emplace_back());
            })) {
            return failure();
        }
    }

    // Parse the before region's `args ( $assignments )`.
    llvm::SmallVector<OpAsmParser::Argument> beforeRegionArgs;
    llvm::SmallVector<OpAsmParser::UnresolvedOperand> beforeOperands;
    if (parser.parseKeyword("args") ||
        parser.parseAssignmentList(beforeRegionArgs, beforeOperands)) {
        return failure();
    }

    if (beforeRegionArgs.size() != types.size()) {
        return parser.emitError(parser.getNameLoc())
               << "expected " << types.size() << " before arguments but got "
               << beforeRegionArgs.size();
    }

    for (auto [arg, ty] : llvm::zip_equal(beforeRegionArgs, types)) {
        arg.type = ty;
    }

    if (parser.parseRegion(*before, beforeRegionArgs)) {
        return failure();
    }
    WhileOp::ensureTerminator(*before, builder, result.location);

    // Parse the after region's `args ( $names )`.
    // Names only. The operands are inherited from the before region's `args(...)`.
    llvm::SmallVector<OpAsmParser::Argument> afterRegionArgs;
    if (parser.parseKeyword("args") ||
        parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
            return parser.parseArgument(afterRegionArgs.emplace_back());
        })) {
        return failure();
    }

    if (afterRegionArgs.size() != types.size()) {
        return parser.emitError(parser.getNameLoc())
               << "expected " << types.size() << " after arguments but got "
               << afterRegionArgs.size();
    }

    for (auto [arg, ty] : llvm::zip_equal(afterRegionArgs, types)) {
        arg.type = ty;
    }

    if (parser.parseRegion(*after, afterRegionArgs)) {
        return failure();
    }
    WhileOp::ensureTerminator(*after, builder, result.location);

    // Resolve operands from the before region's `args(...)`.
    if (parser.resolveOperands(beforeOperands, types, parser.getCurrentLocation(),
                               result.operands)) {
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

    auto beforeArgs = getBefore().getArguments();
    if (verifyRegionArgs(*this, inValues, outValues, beforeArgs).failed()) {
        return failure();
    }

    auto afterArgs = getAfter().getArguments();
    if (verifyRegionArgs(*this, inValues, outValues, afterArgs).failed()) {
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
