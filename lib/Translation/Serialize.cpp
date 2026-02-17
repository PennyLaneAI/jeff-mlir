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
  llvm::SmallVector<mlir::Operation*> operations;
  std::unordered_map<std::string, uint32_t> strings;

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

void serializeMeasure(jeff::Op::Builder opBuilder,
                      mlir::jeff::QubitMeasureOp op,
                      SerializationContext& ctx) {
  auto qubitBuilder = opBuilder.initInstruction().initQubit();
  qubitBuilder.setMeasure();

  auto inputs = opBuilder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getInQubit()));

  auto outputs = opBuilder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getResult()));
}

void serializeMeasureNd(jeff::Op::Builder opBuilder,
                        mlir::jeff::QubitMeasureNDOp op,
                        SerializationContext& ctx) {
  auto qubitBuilder = opBuilder.initInstruction().initQubit();
  qubitBuilder.setMeasureNd();

  auto inputs = opBuilder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getInQubit()));

  auto outputs = opBuilder.initOutputs(2);
  outputs.set(0, ctx.getValueId(op.getOutQubit()));
  outputs.set(1, ctx.getValueId(op.getResult()));
}

void serializeReset(jeff::Op::Builder opBuilder, mlir::jeff::QubitResetOp op,
                    SerializationContext& ctx) {
  auto qubitBuilder = opBuilder.initInstruction().initQubit();
  qubitBuilder.setReset();

  auto inputs = opBuilder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getInQubit()));

  auto outputs = opBuilder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutQubit()));
}

template <typename OpType>
void serializeOneTargetZeroParameter(jeff::Op::Builder opBuilder, OpType op,
                                     jeff::WellKnownGate gate,
                                     SerializationContext& ctx) {
  auto gateBuilder = opBuilder.initInstruction().initQubit().initGate();
  gateBuilder.setWellKnown(gate);
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());

  auto numControls = op.getNumCtrls();
  auto inputs = opBuilder.initInputs(1 + numControls);
  inputs.set(0, ctx.getValueId(op.getInQubit()));
  for (uint8_t i = 0; i < numControls; ++i) {
    inputs.set(1 + i, ctx.getValueId(op.getInCtrlQubits()[i]));
  }

  auto outputs = opBuilder.initOutputs(1 + numControls);
  outputs.set(0, ctx.getValueId(op.getOutQubit()));
  for (uint8_t i = 0; i < numControls; ++i) {
    outputs.set(1 + i, ctx.getValueId(op.getOutCtrlQubits()[i]));
  }
}

template <typename OpType>
void serializeOneTargetOneParameter(jeff::Op::Builder opBuilder, OpType op,
                                    jeff::WellKnownGate gate,
                                    SerializationContext& ctx) {
  auto gateBuilder = opBuilder.initInstruction().initQubit().initGate();
  gateBuilder.setWellKnown(gate);
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());

  auto numControls = op.getNumCtrls();
  auto inputs = opBuilder.initInputs(1 + numControls + 1);
  inputs.set(0, ctx.getValueId(op.getInQubit()));
  for (uint8_t i = 0; i < numControls; ++i) {
    inputs.set(1 + i, ctx.getValueId(op.getInCtrlQubits()[i]));
  }
  inputs.set(1 + numControls, ctx.getValueId(op.getRotation()));

  auto outputs = opBuilder.initOutputs(1 + numControls);
  outputs.set(0, ctx.getValueId(op.getOutQubit()));
  for (uint8_t i = 0; i < numControls; ++i) {
    outputs.set(1 + i, ctx.getValueId(op.getOutCtrlQubits()[i]));
  }
}

void serializeU(jeff::Op::Builder opBuilder, mlir::jeff::UOp op,
                SerializationContext& ctx) {

  auto gateBuilder = opBuilder.initInstruction().initQubit().initGate();
  gateBuilder.setWellKnown(jeff::WellKnownGate::U);
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());

  auto numControls = op.getNumCtrls();
  auto inputs = opBuilder.initInputs(1 + numControls + 3);
  inputs.set(0, ctx.getValueId(op.getInQubit()));
  for (uint8_t i = 0; i < numControls; ++i) {
    inputs.set(1 + i, ctx.getValueId(op.getInCtrlQubits()[i]));
  }
  inputs.set(1 + numControls, ctx.getValueId(op.getTheta()));
  inputs.set(2 + numControls, ctx.getValueId(op.getPhi()));
  inputs.set(3 + numControls, ctx.getValueId(op.getLambda()));

  auto outputs = opBuilder.initOutputs(1 + numControls);
  outputs.set(0, ctx.getValueId(op.getOutQubit()));
  for (uint8_t i = 0; i < numControls; ++i) {
    outputs.set(1 + i, ctx.getValueId(op.getOutCtrlQubits()[i]));
  }
}

void serializeSwap(jeff::Op::Builder opBuilder, mlir::jeff::SwapOp op,
                   SerializationContext& ctx) {
  auto gateBuilder = opBuilder.initInstruction().initQubit().initGate();
  gateBuilder.setWellKnown(jeff::WellKnownGate::SWAP);
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());

  auto numControls = op.getNumCtrls();
  auto inputs = opBuilder.initInputs(2 + numControls);
  inputs.set(0, ctx.getValueId(op.getInQubitOne()));
  inputs.set(1, ctx.getValueId(op.getInQubitTwo()));
  for (uint8_t i = 0; i < numControls; ++i) {
    inputs.set(2 + i, ctx.getValueId(op.getInCtrlQubits()[i]));
  }

  auto outputs = opBuilder.initOutputs(2 + numControls);
  outputs.set(0, ctx.getValueId(op.getOutQubitOne()));
  outputs.set(1, ctx.getValueId(op.getOutQubitTwo()));
  for (uint8_t i = 0; i < numControls; ++i) {
    outputs.set(2 + i, ctx.getValueId(op.getOutCtrlQubits()[i]));
  }
}

void serializeGPhase(jeff::Op::Builder opBuilder, mlir::jeff::GPhaseOp op,
                     SerializationContext& ctx) {
  auto gateBuilder = opBuilder.initInstruction().initQubit().initGate();
  gateBuilder.setWellKnown(jeff::WellKnownGate::GPHASE);
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());

  auto numControls = op.getNumCtrls();
  auto inputs = opBuilder.initInputs(numControls + 1);
  for (uint8_t i = 0; i < numControls; ++i) {
    inputs.set(i, ctx.getValueId(op.getInCtrlQubits()[i]));
  }
  inputs.set(numControls, ctx.getValueId(op.getRotation()));

  auto outputs = opBuilder.initOutputs(numControls);
  for (uint8_t i = 0; i < numControls; ++i) {
    outputs.set(i, ctx.getValueId(op.getOutCtrlQubits()[i]));
  }
}

void serializeCustom(jeff::Op::Builder opBuilder, mlir::jeff::CustomOp op,
                     SerializationContext& ctx) {
  auto gateBuilder = opBuilder.initInstruction().initQubit().initGate();
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());
  auto customBuilder = gateBuilder.initCustom();
  customBuilder.setName(ctx.strings[op.getName().str()]);
  customBuilder.setNumQubits(op.getNumTargets());
  customBuilder.setNumParams(op.getNumParams());

  auto inputs = opBuilder.initInputs(op.getNumOperands());
  for (size_t i = 0; i < op.getNumOperands(); ++i) {
    inputs.set(i, ctx.getValueId(op.getOperand(i)));
  }

  auto outputs = opBuilder.initOutputs(op.getNumResults());
  for (size_t i = 0; i < op.getNumResults(); ++i) {
    outputs.set(i, ctx.getValueId(op.getResult(i)));
  }
}

void serializePpr(jeff::Op::Builder opBuilder, mlir::jeff::PPROp op,
                  SerializationContext& ctx) {
  auto gateBuilder = opBuilder.initInstruction().initQubit().initGate();
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());
  auto pprBuilder = gateBuilder.initPpr();

  auto pauliString = op.getPauliGates();
  capnp::List<jeff::Pauli>::Builder pauliStringBuilder =
      pprBuilder.initPauliString(pauliString.size());
  for (size_t i = 0; i < pauliString.size(); ++i) {
    switch (pauliString[i]) {
    case 0:
      pauliStringBuilder.set(i, jeff::Pauli::I);
      break;
    case 1:
      pauliStringBuilder.set(i, jeff::Pauli::X);
      break;
    case 2:
      pauliStringBuilder.set(i, jeff::Pauli::Y);
      break;
    case 3:
      pauliStringBuilder.set(i, jeff::Pauli::Z);
      break;
    default:
      llvm::report_fatal_error("Unknown Pauli gate");
    }
  }

  auto inputs = opBuilder.initInputs(op.getNumOperands());
  for (size_t i = 0; i < op.getNumOperands(); ++i) {
    inputs.set(i, ctx.getValueId(op.getOperand(i)));
  }

  auto outputs = opBuilder.initOutputs(op.getNumResults());
  for (size_t i = 0; i < op.getNumResults(); ++i) {
    outputs.set(i, ctx.getValueId(op.getResult(i)));
  }
}

//===----------------------------------------------------------------------===//
// Float operations
//===----------------------------------------------------------------------===//

void serializeFloatConst64(jeff::Op::Builder opBuilder,
                           mlir::jeff::FloatConst64Op op,
                           SerializationContext& ctx) {
  auto floatBuilder = opBuilder.initInstruction().initFloat();
  auto value = op.getVal().convertToDouble();
  floatBuilder.setConst64(value);

  opBuilder.initInputs(0);

  auto outputs = opBuilder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getConstant()));
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
    serializeMeasure(opBuilder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::QubitMeasureNDOp>(operation)) {
    serializeMeasureNd(opBuilder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::QubitResetOp>(operation)) {
    serializeReset(opBuilder, op, ctx);
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
  } else if (auto op = llvm::dyn_cast<mlir::jeff::R1Op>(operation)) {
    serializeOneTargetOneParameter(opBuilder, op, jeff::WellKnownGate::R1, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::RxOp>(operation)) {
    serializeOneTargetOneParameter(opBuilder, op, jeff::WellKnownGate::RX, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::RyOp>(operation)) {
    serializeOneTargetOneParameter(opBuilder, op, jeff::WellKnownGate::RY, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::RzOp>(operation)) {
    serializeOneTargetOneParameter(opBuilder, op, jeff::WellKnownGate::RZ, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::HOp>(operation)) {
    serializeOneTargetZeroParameter(opBuilder, op, jeff::WellKnownGate::H, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::UOp>(operation)) {
    serializeU(opBuilder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::SwapOp>(operation)) {
    serializeSwap(opBuilder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::IOp>(operation)) {
    serializeOneTargetZeroParameter(opBuilder, op, jeff::WellKnownGate::I, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::GPhaseOp>(operation)) {
    serializeGPhase(opBuilder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::CustomOp>(operation)) {
    serializeCustom(opBuilder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::PPROp>(operation)) {
    serializePpr(opBuilder, op, ctx);
  }
  // Float operations
  else if (auto op = llvm::dyn_cast<mlir::jeff::FloatConst64Op>(operation)) {
    serializeFloatConst64(opBuilder, op, ctx);
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

  // Create capnp message
  capnp::MallocMessageBuilder message;
  auto moduleBuilder = message.initRoot<jeff::Module>();

  // Get strings
  auto stringsAttr =
      llvm::cast<mlir::ArrayAttr>(module->getAttr("jeff.strings"));
  auto stringsBuilder = moduleBuilder.initStrings(stringsAttr.size());
  for (auto i = 0; i < stringsAttr.size(); ++i) {
    auto str = llvm::cast<mlir::StringAttr>(stringsAttr[i]).getValue().str();
    ctx.strings[str] = i;
    stringsBuilder.set(i, str);
  }

  // Get functions
  llvm::SmallVector<mlir::func::FuncOp> functions;
  module.walk([&](mlir::func::FuncOp func) { functions.push_back(func); });

  // Build functions
  auto functionsBuilder = moduleBuilder.initFunctions(functions.size());

  // TODO: Support multiple functions
  auto function = functions[0];
  auto funcBuilder = functionsBuilder[0];
  serializeFunction(funcBuilder, function,
                    ctx.strings[function.getName().str()], ctx);

  // Set metadata
  moduleBuilder.setTool("");
  moduleBuilder.setToolVersion("");

  return capnp::messageToFlatArray(message);
}
