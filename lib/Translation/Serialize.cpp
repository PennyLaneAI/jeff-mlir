#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffOps.h"

#include <capnp/common.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <cstddef>
#include <cstdint>
#include <jeff.capnp.h>
#include <kj/array.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <string>
#include <unordered_map>

namespace {

struct SerializationContext {
  llvm::DenseMap<mlir::Value, uint32_t> valueMap;
  llvm::DenseMap<mlir::func::FuncOp, uint32_t> funcMap;
  std::unordered_map<std::string, uint32_t> stringMap;
  llvm::SmallVector<std::string> strings;
  llvm::SmallVector<mlir::Operation*> operations;

  uint32_t getOrAddString(const std::string& str) {
    auto it = stringMap.find(str);
    if (it != stringMap.end()) {
      return it->second;
    }
    uint32_t index = strings.size();
    strings.push_back(str);
    stringMap[str] = index;
    return index;
  }

  uint32_t getValueId(mlir::Value value) {
    auto it = valueMap.find(value);
    if (it != valueMap.end()) {
      return it->second;
    }
    uint32_t id = valueMap.size();
    valueMap[value] = id;
    return id;
  }
};

//===----------------------------------------------------------------------===//
// Qubit operations
//===----------------------------------------------------------------------===//

void serializeQubitAlloc(jeff::Op::Builder opBuilder,
                         mlir::jeff::QubitAllocOp op,
                         SerializationContext& ctx) {
  auto qubitBuilder = opBuilder.initInstruction().initQubit();
  qubitBuilder.setAlloc();

  opBuilder.initInputs(0);

  auto outputs = opBuilder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getResult()));
}

void serializeQubitFree(jeff::Op::Builder opBuilder, mlir::jeff::QubitFreeOp op,
                        SerializationContext& ctx) {
  auto qubitBuilder = opBuilder.initInstruction().initQubit();
  qubitBuilder.setFree();

  auto inputs = opBuilder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getInQubit()));

  opBuilder.initOutputs(0);
}

void serializeQubitFreeZero(jeff::Op::Builder opBuilder,
                            mlir::jeff::QubitFreeZeroOp op,
                            SerializationContext& ctx) {
  auto qubitBuilder = opBuilder.initInstruction().initQubit();
  qubitBuilder.setFreeZero();

  auto inputs = opBuilder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getInQubit()));

  opBuilder.initOutputs(0);
}

void serializeQubitMeasure(jeff::Op::Builder opBuilder,
                           mlir::jeff::QubitMeasureOp op,
                           SerializationContext& ctx) {
  auto qubitBuilder = opBuilder.initInstruction().initQubit();
  qubitBuilder.setMeasure();

  auto inputs = opBuilder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getInQubit()));

  auto outputs = opBuilder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getResult()));
}

template <typename OpType>
void serializeOneTargetZeroParameter(jeff::Op::Builder opBuilder, OpType op,
                                     jeff::WellKnownGate gate,
                                     SerializationContext& ctx) {
  auto qubitBuilder = opBuilder.initInstruction().initQubit();
  auto gateBuilder = qubitBuilder.initGate();
  gateBuilder.setWellKnown(gate);
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());

  uint8_t numCtrls = op.getNumCtrls();
  auto inputs = opBuilder.initInputs(1 + numCtrls);
  inputs.set(0, ctx.getValueId(op.getInQubit()));
  for (uint8_t i = 0; i < numCtrls; ++i) {
    inputs.set(1 + i, ctx.getValueId(op.getInCtrlQubits()[i]));
  }

  auto outputs = opBuilder.initOutputs(1 + numCtrls);
  outputs.set(0, ctx.getValueId(op.getOutQubit()));
  for (uint8_t i = 0; i < numCtrls; ++i) {
    outputs.set(1 + i, ctx.getValueId(op.getOutCtrlQubits()[i]));
  }
}

template <typename OpType>
void serializeOneTargetOneParameter(jeff::Op::Builder opBuilder, OpType op,
                                    jeff::WellKnownGate gate,
                                    SerializationContext& ctx) {
  auto qubitBuilder = opBuilder.initInstruction().initQubit();
  auto gateBuilder = qubitBuilder.initGate();
  gateBuilder.setWellKnown(gate);
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());

  uint8_t numCtrls = op.getNumCtrls();
  auto inputs = opBuilder.initInputs(1 + numCtrls + 1);
  inputs.set(0, ctx.getValueId(op.getInQubit()));
  for (uint8_t i = 0; i < numCtrls; ++i) {
    inputs.set(1 + i, ctx.getValueId(op.getInCtrlQubits()[i]));
  }
  inputs.set(1 + numCtrls, ctx.getValueId(op.getRotation()));

  auto outputs = opBuilder.initOutputs(1 + numCtrls);
  outputs.set(0, ctx.getValueId(op.getOutQubit()));
  for (uint8_t i = 0; i < numCtrls; ++i) {
    outputs.set(1 + i, ctx.getValueId(op.getOutCtrlQubits()[i]));
  }
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

void serializeQubitType(jeff::Type::Builder builder) { builder.setQubit(); }

void serializeQuregType(jeff::Type::Builder builder) { builder.setQureg(); }

void serializeIntType(jeff::Type::Builder builder, mlir::Type type) {
  auto intType = llvm::cast<mlir::IntegerType>(type);
  unsigned width = intType.getWidth();
  builder.setInt(width);
}

void serializeIntArrayType(jeff::Type::Builder builder, mlir::Type type) {
  auto tensorType = llvm::cast<mlir::RankedTensorType>(type);
  auto elementType = llvm::cast<mlir::IntegerType>(tensorType.getElementType());
  unsigned width = elementType.getWidth();
  builder.setIntArray(width);
}

void serializeFloatType(jeff::Type::Builder builder, mlir::Type type) {
  auto floatType = llvm::cast<mlir::FloatType>(type);
  if (floatType.getWidth() == 32) {
    builder.setFloat(jeff::FloatPrecision::FLOAT32);
  } else if (floatType.getWidth() == 64) {
    builder.setFloat(jeff::FloatPrecision::FLOAT64);
  } else {
    llvm::errs() << "Cannot serialize floats with bit width "
                 << floatType.getWidth() << "\n";
    llvm::report_fatal_error("Unknown float type");
  }
}

void serializeFloatArrayType(jeff::Type::Builder builder, mlir::Type type) {
  auto tensorType = llvm::cast<mlir::RankedTensorType>(type);
  auto elementType = llvm::cast<mlir::FloatType>(tensorType.getElementType());
  if (elementType.getWidth() == 32) {
    builder.setFloatArray(jeff::FloatPrecision::FLOAT32);
  } else if (elementType.getWidth() == 64) {
    builder.setFloatArray(jeff::FloatPrecision::FLOAT64);
  } else {
    llvm::errs() << "Cannot serialize float arrays with bit width "
                 << elementType.getWidth() << "\n";
    llvm::report_fatal_error("Unknown float array type");
  }
}

void serializeType(jeff::Type::Builder builder, mlir::Type type) {
  if (llvm::isa<mlir::jeff::QubitType>(type)) {
    serializeQubitType(builder);
  } else if (llvm::isa<mlir::jeff::QuregType>(type)) {
    serializeQuregType(builder);
  } else if (llvm::isa<mlir::IntegerType>(type)) {
    serializeIntType(builder, type);
  } else if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
    if (llvm::isa<mlir::IntegerType>(tensorType.getElementType())) {
      serializeIntArrayType(builder, type);
    } else if (llvm::isa<mlir::FloatType>(tensorType.getElementType())) {
      serializeFloatArrayType(builder, type);
    } else {
      llvm::report_fatal_error("Unknown tensor element type");
    }
  } else if (llvm::isa<mlir::FloatType>(type)) {
    serializeFloatType(builder, type);
  } else {
    llvm::report_fatal_error("Unknown type");
  }
}

void serializeOperation(jeff::Op::Builder opBuilder, mlir::Operation* operation,
                        SerializationContext& ctx) {
  // Qubit operations
  if (auto op = llvm::dyn_cast<mlir::jeff::QubitAllocOp>(operation)) {
    serializeQubitAlloc(opBuilder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::QubitFreeOp>(operation)) {
    serializeQubitFree(opBuilder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::QubitFreeZeroOp>(operation)) {
    serializeQubitFreeZero(opBuilder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::QubitMeasureOp>(operation)) {
    serializeQubitMeasure(opBuilder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::HOp>(operation)) {
    serializeOneTargetZeroParameter(opBuilder, op, jeff::WellKnownGate::H, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::XOp>(operation)) {
    serializeOneTargetZeroParameter(opBuilder, op, jeff::WellKnownGate::X, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::YOp>(operation)) {
    serializeOneTargetZeroParameter(opBuilder, op, jeff::WellKnownGate::Y, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::ZOp>(operation)) {
    serializeOneTargetZeroParameter(opBuilder, op, jeff::WellKnownGate::Z, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::SOp>(operation)) {
    serializeOneTargetZeroParameter(opBuilder, op, jeff::WellKnownGate::S, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::TOp>(operation)) {
    serializeOneTargetZeroParameter(opBuilder, op, jeff::WellKnownGate::T, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::IOp>(operation)) {
    serializeOneTargetZeroParameter(opBuilder, op, jeff::WellKnownGate::I, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::RxOp>(operation)) {
    serializeOneTargetOneParameter(opBuilder, op, jeff::WellKnownGate::RX, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::RyOp>(operation)) {
    serializeOneTargetOneParameter(opBuilder, op, jeff::WellKnownGate::RY, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::RzOp>(operation)) {
    serializeOneTargetOneParameter(opBuilder, op, jeff::WellKnownGate::RZ, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::R1Op>(operation)) {
    serializeOneTargetOneParameter(opBuilder, op, jeff::WellKnownGate::R1, ctx);
  } else {
    llvm::errs() << "Cannot serialize operation " << operation->getName()
                 << "\n";
    llvm::report_fatal_error("Unknown operation");
  }
}

void collectValues(mlir::func::FuncOp func, SerializationContext& ctx,
                   llvm::SmallVector<mlir::Value>& values) {
  // Collect all values: block arguments and operation results
  for (auto& block : func.getRegion()) {
    for (auto arg : block.getArguments()) {
      uint32_t id = ctx.getValueId(arg);
      if (id >= values.size()) {
        values.resize(id + 1);
      }
      values[id] = arg;
    }
    for (auto& op : block) {
      for (auto result : op.getResults()) {
        uint32_t id = ctx.getValueId(result);
        if (id >= values.size()) {
          values.resize(id + 1);
        }
        values[id] = result;
      }
    }
  }
}

void serializeFunction(jeff::Function::Builder funcBuilder,
                       mlir::func::FuncOp func, uint32_t funcNameIdx,
                       SerializationContext& ctx) {
  funcBuilder.setName(funcNameIdx);

  auto defBuilder = funcBuilder.initDefinition();
  auto& entryBlock = func.getRegion().front();

  // Collect all operations (except return)
  llvm::SmallVector<mlir::Operation*> ops;
  for (auto& op : entryBlock) {
    if (!llvm::isa<mlir::func::ReturnOp>(&op)) {
      ops.push_back(&op);
    }
  }

  // Collect all values
  llvm::SmallVector<mlir::Value> values;
  collectValues(func, ctx, values);

  // Build values list
  auto valuesBuilder = defBuilder.initValues(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    auto valueBuilder = valuesBuilder[i];
    auto typeBuilder = valueBuilder.initType();
    serializeType(typeBuilder, values[i].getType());
  }

  // Build body
  auto bodyBuilder = defBuilder.initBody();

  // Set sources (function arguments)
  auto numArgs = entryBlock.getNumArguments();
  auto sourcesBuilder = bodyBuilder.initSources(numArgs);
  for (unsigned i = 0; i < numArgs; ++i) {
    sourcesBuilder.set(i, ctx.getValueId(entryBlock.getArgument(i)));
  }

  // Set targets (return values)
  mlir::func::ReturnOp returnOp;
  for (auto& op : entryBlock) {
    if (auto ret = llvm::dyn_cast<mlir::func::ReturnOp>(&op)) {
      returnOp = ret;
      break;
    }
  }

  auto targetsBuilder = bodyBuilder.initTargets(returnOp.getNumOperands());
  for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
    targetsBuilder.set(i, ctx.getValueId(returnOp.getOperand(i)));
  }

  // Serialize operations
  auto opsBuilder = bodyBuilder.initOperations(ops.size());
  for (size_t i = 0; i < ops.size(); ++i) {
    auto opBuilder = opsBuilder[i];
    serializeOperation(opBuilder, ops[i], ctx);
  }
}

} // namespace

kj::Array<capnp::word> serialize(mlir::ModuleOp module) {
  SerializationContext ctx;

  // Collect all functions
  llvm::SmallVector<mlir::func::FuncOp> functions;
  module.walk([&](mlir::func::FuncOp func) { functions.push_back(func); });

  // Create capnp message
  capnp::MallocMessageBuilder message;
  auto moduleBuilder = message.initRoot<jeff::Module>();

  // Set version
  moduleBuilder.setVersion(0);

  // Build functions
  auto functionsBuilder = moduleBuilder.initFunctions(functions.size());

  uint32_t entryPointIdx = 0;
  bool foundEntryPoint = false;

  for (size_t i = 0; i < functions.size(); ++i) {
    auto func = functions[i];
    auto funcName = func.getName().str();
    uint32_t funcNameIdx = ctx.getOrAddString(funcName);

    auto funcBuilder = functionsBuilder[i];
    serializeFunction(funcBuilder, func, funcNameIdx, ctx);
  }

  // Set entry point
  if (foundEntryPoint) {
    moduleBuilder.setEntrypoint(entryPointIdx);
  }

  moduleBuilder.setTool("");
  moduleBuilder.setToolVersion("");

  // Build strings
  auto stringsBuilder = moduleBuilder.initStrings(ctx.strings.size());
  for (size_t i = 0; i < ctx.strings.size(); ++i) {
    stringsBuilder.set(i, ctx.strings[i]);
  }

  return capnp::messageToFlatArray(message);
}
