#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffInterfaces.h"
#include "jeff/IR/JeffOps.h"

#include <capnp/common.h>
#include <capnp/list.h>
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
  llvm::DenseMap<mlir::Value, uint32_t> values;
  llvm::DenseMap<llvm::StringRef, uint32_t> funcs;
  std::unordered_map<std::string, uint32_t> strings;

  uint32_t getValueId(mlir::Value value) {
    auto it = values.find(value);
    if (it != values.end()) {
      return it->second;
    }
    auto id = values.size();
    values[value] = id;
    return id;
  }

  uint32_t getFuncId(llvm::StringRef funcName) {
    auto it = funcs.find(funcName);
    if (it == funcs.end()) {
      llvm::errs() << "Function " << funcName << " not found\n";
      llvm::report_fatal_error("Function not found");
    }
    return it->second;
  }

  uint32_t getStringId(llvm::StringRef str) {
    auto it = strings.find(str.str());
    if (it == strings.end()) {
      llvm::errs() << "String " << str << " not found\n";
      llvm::report_fatal_error("String not found");
    }
    return it->second;
  }
};

//===----------------------------------------------------------------------===//
// Qubit operations
//===----------------------------------------------------------------------===//

void serializeQubitAlloc(jeff::Op::Builder builder, mlir::jeff::QubitAllocOp op,
                         SerializationContext& ctx) {
  auto qubitBuilder = builder.initInstruction().initQubit();
  qubitBuilder.setAlloc();

  builder.initInputs(0);

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getResult()));
}

void serializeQubitFree(jeff::Op::Builder builder, mlir::jeff::QubitFreeOp op,
                        SerializationContext& ctx) {
  auto qubitBuilder = builder.initInstruction().initQubit();
  qubitBuilder.setFree();

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getInQubit()));

  builder.initOutputs(0);
}

void serializeQubitFreeZero(jeff::Op::Builder builder,
                            mlir::jeff::QubitFreeZeroOp op,
                            SerializationContext& ctx) {
  auto qubitBuilder = builder.initInstruction().initQubit();
  qubitBuilder.setFreeZero();

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getInQubit()));

  builder.initOutputs(0);
}

void serializeMeasure(jeff::Op::Builder builder, mlir::jeff::QubitMeasureOp op,
                      SerializationContext& ctx) {
  auto qubitBuilder = builder.initInstruction().initQubit();
  qubitBuilder.setMeasure();

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getInQubit()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getResult()));
}

void serializeMeasureNd(jeff::Op::Builder builder,
                        mlir::jeff::QubitMeasureNDOp op,
                        SerializationContext& ctx) {
  auto qubitBuilder = builder.initInstruction().initQubit();
  qubitBuilder.setMeasureNd();

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getInQubit()));

  auto outputs = builder.initOutputs(2);
  outputs.set(0, ctx.getValueId(op.getOutQubit()));
  outputs.set(1, ctx.getValueId(op.getResult()));
}

void serializeReset(jeff::Op::Builder builder, mlir::jeff::QubitResetOp op,
                    SerializationContext& ctx) {
  auto qubitBuilder = builder.initInstruction().initQubit();
  qubitBuilder.setReset();

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getInQubit()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutQubit()));
}

template <typename OpType>
void serializeOneTargetZeroParameter(jeff::Op::Builder builder, OpType op,
                                     jeff::WellKnownGate gate,
                                     SerializationContext& ctx) {
  auto gateBuilder = builder.initInstruction().initQubit().initGate();
  gateBuilder.setWellKnown(gate);
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());

  auto numControls = op.getNumCtrls();
  auto inputs = builder.initInputs(1 + numControls);
  inputs.set(0, ctx.getValueId(op.getInQubit()));
  for (uint8_t i = 0; i < numControls; ++i) {
    inputs.set(1 + i, ctx.getValueId(op.getInCtrlQubits()[i]));
  }

  auto outputs = builder.initOutputs(1 + numControls);
  outputs.set(0, ctx.getValueId(op.getOutQubit()));
  for (uint8_t i = 0; i < numControls; ++i) {
    outputs.set(1 + i, ctx.getValueId(op.getOutCtrlQubits()[i]));
  }
}

template <typename OpType>
void serializeOneTargetOneParameter(jeff::Op::Builder builder, OpType op,
                                    jeff::WellKnownGate gate,
                                    SerializationContext& ctx) {
  auto gateBuilder = builder.initInstruction().initQubit().initGate();
  gateBuilder.setWellKnown(gate);
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());

  auto numControls = op.getNumCtrls();
  auto inputs = builder.initInputs(1 + numControls + 1);
  inputs.set(0, ctx.getValueId(op.getInQubit()));
  for (uint8_t i = 0; i < numControls; ++i) {
    inputs.set(1 + i, ctx.getValueId(op.getInCtrlQubits()[i]));
  }
  inputs.set(1 + numControls, ctx.getValueId(op.getRotation()));

  auto outputs = builder.initOutputs(1 + numControls);
  outputs.set(0, ctx.getValueId(op.getOutQubit()));
  for (uint8_t i = 0; i < numControls; ++i) {
    outputs.set(1 + i, ctx.getValueId(op.getOutCtrlQubits()[i]));
  }
}

void serializeU(jeff::Op::Builder builder, mlir::jeff::UOp op,
                SerializationContext& ctx) {

  auto gateBuilder = builder.initInstruction().initQubit().initGate();
  gateBuilder.setWellKnown(jeff::WellKnownGate::U);
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());

  auto numControls = op.getNumCtrls();
  auto inputs = builder.initInputs(1 + numControls + 3);
  inputs.set(0, ctx.getValueId(op.getInQubit()));
  for (uint8_t i = 0; i < numControls; ++i) {
    inputs.set(1 + i, ctx.getValueId(op.getInCtrlQubits()[i]));
  }
  inputs.set(1 + numControls, ctx.getValueId(op.getTheta()));
  inputs.set(2 + numControls, ctx.getValueId(op.getPhi()));
  inputs.set(3 + numControls, ctx.getValueId(op.getLambda()));

  auto outputs = builder.initOutputs(1 + numControls);
  outputs.set(0, ctx.getValueId(op.getOutQubit()));
  for (uint8_t i = 0; i < numControls; ++i) {
    outputs.set(1 + i, ctx.getValueId(op.getOutCtrlQubits()[i]));
  }
}

void serializeSwap(jeff::Op::Builder builder, mlir::jeff::SwapOp op,
                   SerializationContext& ctx) {
  auto gateBuilder = builder.initInstruction().initQubit().initGate();
  gateBuilder.setWellKnown(jeff::WellKnownGate::SWAP);
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());

  auto numControls = op.getNumCtrls();
  auto inputs = builder.initInputs(2 + numControls);
  inputs.set(0, ctx.getValueId(op.getInQubitOne()));
  inputs.set(1, ctx.getValueId(op.getInQubitTwo()));
  for (uint8_t i = 0; i < numControls; ++i) {
    inputs.set(2 + i, ctx.getValueId(op.getInCtrlQubits()[i]));
  }

  auto outputs = builder.initOutputs(2 + numControls);
  outputs.set(0, ctx.getValueId(op.getOutQubitOne()));
  outputs.set(1, ctx.getValueId(op.getOutQubitTwo()));
  for (uint8_t i = 0; i < numControls; ++i) {
    outputs.set(2 + i, ctx.getValueId(op.getOutCtrlQubits()[i]));
  }
}

void serializeGPhase(jeff::Op::Builder builder, mlir::jeff::GPhaseOp op,
                     SerializationContext& ctx) {
  auto gateBuilder = builder.initInstruction().initQubit().initGate();
  gateBuilder.setWellKnown(jeff::WellKnownGate::GPHASE);
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());

  auto numControls = op.getNumCtrls();
  auto inputs = builder.initInputs(numControls + 1);
  for (uint8_t i = 0; i < numControls; ++i) {
    inputs.set(i, ctx.getValueId(op.getInCtrlQubits()[i]));
  }
  inputs.set(numControls, ctx.getValueId(op.getRotation()));

  auto outputs = builder.initOutputs(numControls);
  for (uint8_t i = 0; i < numControls; ++i) {
    outputs.set(i, ctx.getValueId(op.getOutCtrlQubits()[i]));
  }
}

void serializeWellKnownGate(jeff::Op::Builder builder,
                            mlir::Operation* operation,
                            SerializationContext& ctx) {
  if (auto op = llvm::dyn_cast<mlir::jeff::XOp>(operation)) {
    serializeOneTargetZeroParameter(builder, op, jeff::WellKnownGate::X, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::YOp>(operation)) {
    serializeOneTargetZeroParameter(builder, op, jeff::WellKnownGate::Y, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::ZOp>(operation)) {
    serializeOneTargetZeroParameter(builder, op, jeff::WellKnownGate::Z, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::SOp>(operation)) {
    serializeOneTargetZeroParameter(builder, op, jeff::WellKnownGate::S, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::TOp>(operation)) {
    serializeOneTargetZeroParameter(builder, op, jeff::WellKnownGate::T, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::R1Op>(operation)) {
    serializeOneTargetOneParameter(builder, op, jeff::WellKnownGate::R1, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::RxOp>(operation)) {
    serializeOneTargetOneParameter(builder, op, jeff::WellKnownGate::RX, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::RyOp>(operation)) {
    serializeOneTargetOneParameter(builder, op, jeff::WellKnownGate::RY, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::RzOp>(operation)) {
    serializeOneTargetOneParameter(builder, op, jeff::WellKnownGate::RZ, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::HOp>(operation)) {
    serializeOneTargetZeroParameter(builder, op, jeff::WellKnownGate::H, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::UOp>(operation)) {
    serializeU(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::SwapOp>(operation)) {
    serializeSwap(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::IOp>(operation)) {
    serializeOneTargetZeroParameter(builder, op, jeff::WellKnownGate::I, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::GPhaseOp>(operation)) {
    serializeGPhase(builder, op, ctx);
  } else {
    llvm::errs() << "Cannot serialize well-known gate " << operation->getName()
                 << "\n";
    llvm::report_fatal_error("Unknown well-known gate");
  }
}

void serializeCustom(jeff::Op::Builder builder, mlir::jeff::CustomOp op,
                     SerializationContext& ctx) {
  auto gateBuilder = builder.initInstruction().initQubit().initGate();
  gateBuilder.setControlQubits(op.getNumCtrls());
  gateBuilder.setAdjoint(op.getIsAdjoint());
  gateBuilder.setPower(op.getPower());
  auto customBuilder = gateBuilder.initCustom();
  customBuilder.setName(ctx.getStringId(op.getName().str()));
  customBuilder.setNumQubits(op.getNumTargets());
  customBuilder.setNumParams(op.getNumParams());

  auto inputs = builder.initInputs(op.getNumOperands());
  for (size_t i = 0; i < op.getNumOperands(); ++i) {
    inputs.set(i, ctx.getValueId(op.getOperand(i)));
  }

  auto outputs = builder.initOutputs(op.getNumResults());
  for (size_t i = 0; i < op.getNumResults(); ++i) {
    outputs.set(i, ctx.getValueId(op.getResult(i)));
  }
}

void serializePpr(jeff::Op::Builder builder, mlir::jeff::PPROp op,
                  SerializationContext& ctx) {
  auto gateBuilder = builder.initInstruction().initQubit().initGate();
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

  auto inputs = builder.initInputs(op.getNumOperands());
  for (size_t i = 0; i < op.getNumOperands(); ++i) {
    inputs.set(i, ctx.getValueId(op.getOperand(i)));
  }

  auto outputs = builder.initOutputs(op.getNumResults());
  for (size_t i = 0; i < op.getNumResults(); ++i) {
    outputs.set(i, ctx.getValueId(op.getResult(i)));
  }
}

void serializeGate(jeff::Op::Builder builder, mlir::Operation* operation,
                   SerializationContext& ctx) {
  if (llvm::isa<mlir::jeff::WellKnownGate>(operation)) {
    serializeWellKnownGate(builder, operation, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::CustomOp>(operation)) {
    serializeCustom(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::PPROp>(operation)) {
    serializePpr(builder, op, ctx);
  } else {
    llvm::errs() << "Cannot serialize gate operation " << operation->getName()
                 << "\n";
    llvm::report_fatal_error("Unknown gate operation");
  }
}

void serializeQubit(jeff::Op::Builder builder, mlir::Operation* operation,
                    SerializationContext& ctx) {
  if (auto op = llvm::dyn_cast<mlir::jeff::QubitAllocOp>(operation)) {
    serializeQubitAlloc(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::QubitFreeOp>(operation)) {
    serializeQubitFree(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::QubitFreeZeroOp>(operation)) {
    serializeQubitFreeZero(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::QubitMeasureOp>(operation)) {
    serializeMeasure(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::QubitMeasureNDOp>(operation)) {
    serializeMeasureNd(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::QubitResetOp>(operation)) {
    serializeReset(builder, op, ctx);
  } else if (llvm::isa<mlir::jeff::MultipleCtrlQubitsGate>(operation)) {
    serializeGate(builder, operation, ctx);
  } else {
    llvm::errs() << "Cannot serialize qubit operation " << operation->getName()
                 << "\n";
    llvm::report_fatal_error("Unknown qubit operation");
  }
}

//===----------------------------------------------------------------------===//
// Qureg operations
//===----------------------------------------------------------------------===//

void serializeQuregAlloc(jeff::Op::Builder builder, mlir::jeff::QuregAllocOp op,
                         SerializationContext& ctx) {
  auto quregBuilder = builder.initInstruction().initQureg();
  quregBuilder.setAlloc();

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getNumQubits()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getResult()));
}

void serializeQuregFreeZero(jeff::Op::Builder builder,
                            mlir::jeff::QuregFreeZeroOp op,
                            SerializationContext& ctx) {
  auto quregBuilder = builder.initInstruction().initQureg();
  quregBuilder.setFreeZero();

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getQreg()));

  builder.initOutputs(0);
}

void serializeQuregExtractIndex(jeff::Op::Builder builder,
                                mlir::jeff::QuregExtractIndexOp op,
                                SerializationContext& ctx) {
  auto quregBuilder = builder.initInstruction().initQureg();
  quregBuilder.setExtractIndex();

  auto inputs = builder.initInputs(2);
  inputs.set(0, ctx.getValueId(op.getInQreg()));
  inputs.set(1, ctx.getValueId(op.getIndex()));

  auto outputs = builder.initOutputs(2);
  outputs.set(0, ctx.getValueId(op.getOutQreg()));
  outputs.set(1, ctx.getValueId(op.getOutQubit()));
}

void serializeQuregInsertIndex(jeff::Op::Builder builder,
                               mlir::jeff::QuregInsertIndexOp op,
                               SerializationContext& ctx) {
  auto quregBuilder = builder.initInstruction().initQureg();
  quregBuilder.setInsertIndex();

  auto inputs = builder.initInputs(3);
  inputs.set(0, ctx.getValueId(op.getInQreg()));
  inputs.set(1, ctx.getValueId(op.getInQubit()));
  inputs.set(2, ctx.getValueId(op.getIndex()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutQreg()));
}

void serializeQuregExtractSlice(jeff::Op::Builder builder,
                                mlir::jeff::QuregExtractSliceOp op,
                                SerializationContext& ctx) {
  auto quregBuilder = builder.initInstruction().initQureg();
  quregBuilder.setExtractSlice();

  auto inputs = builder.initInputs(3);
  inputs.set(0, ctx.getValueId(op.getInQreg()));
  inputs.set(1, ctx.getValueId(op.getStart()));
  inputs.set(2, ctx.getValueId(op.getLength()));

  auto outputs = builder.initOutputs(2);
  outputs.set(0, ctx.getValueId(op.getOutQreg()));
  outputs.set(1, ctx.getValueId(op.getNewQreg()));
}

void serializeQuregInsertSlice(jeff::Op::Builder builder,
                               mlir::jeff::QuregInsertSliceOp op,
                               SerializationContext& ctx) {
  auto quregBuilder = builder.initInstruction().initQureg();
  quregBuilder.setInsertSlice();

  auto inputs = builder.initInputs(3);
  inputs.set(0, ctx.getValueId(op.getInQreg()));
  inputs.set(1, ctx.getValueId(op.getNewQreg()));
  inputs.set(2, ctx.getValueId(op.getStart()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutQreg()));
}

void serializeQuregLength(jeff::Op::Builder builder,
                          mlir::jeff::QuregLengthOp op,
                          SerializationContext& ctx) {
  auto quregBuilder = builder.initInstruction().initQureg();
  quregBuilder.setLength();

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getInQreg()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutQreg()));
}

void serializeQuregSplit(jeff::Op::Builder builder, mlir::jeff::QuregSplitOp op,
                         SerializationContext& ctx) {
  auto quregBuilder = builder.initInstruction().initQureg();
  quregBuilder.setSplit();

  auto inputs = builder.initInputs(2);
  inputs.set(0, ctx.getValueId(op.getInQreg()));
  inputs.set(1, ctx.getValueId(op.getIndex()));

  auto outputs = builder.initOutputs(2);
  outputs.set(0, ctx.getValueId(op.getOutQregOne()));
  outputs.set(1, ctx.getValueId(op.getOutQregTwo()));
}

void serializeQuregJoin(jeff::Op::Builder builder, mlir::jeff::QuregJoinOp op,
                        SerializationContext& ctx) {
  auto quregBuilder = builder.initInstruction().initQureg();
  quregBuilder.setJoin();

  auto inputs = builder.initInputs(2);
  inputs.set(0, ctx.getValueId(op.getInQregOne()));
  inputs.set(1, ctx.getValueId(op.getInQregTwo()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutQreg()));
}

void serializeQuregCreate(jeff::Op::Builder builder,
                          mlir::jeff::QuregCreateOp op,
                          SerializationContext& ctx) {
  auto quregBuilder = builder.initInstruction().initQureg();
  quregBuilder.setCreate();

  auto inputs = builder.initInputs(op.getInQubits().size());
  for (size_t i = 0; i < op.getInQubits().size(); ++i) {
    inputs.set(i, ctx.getValueId(op.getInQubits()[i]));
  }

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutQreg()));
}

void serializeQuregFree(jeff::Op::Builder builder, mlir::jeff::QuregFreeOp op,
                        SerializationContext& ctx) {
  auto quregBuilder = builder.initInstruction().initQureg();
  quregBuilder.setFree();

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getQreg()));

  builder.initOutputs(0);
}

void serializeQureg(jeff::Op::Builder builder, mlir::Operation* operation,
                    SerializationContext& ctx) {
  if (auto op = llvm::dyn_cast<mlir::jeff::QuregAllocOp>(operation)) {
    serializeQuregAlloc(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::QuregFreeZeroOp>(operation)) {
    serializeQuregFreeZero(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::QuregExtractIndexOp>(operation)) {
    serializeQuregExtractIndex(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::QuregInsertIndexOp>(operation)) {
    serializeQuregInsertIndex(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::QuregExtractSliceOp>(operation)) {
    serializeQuregExtractSlice(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::QuregInsertSliceOp>(operation)) {
    serializeQuregInsertSlice(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::QuregLengthOp>(operation)) {
    serializeQuregLength(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::QuregSplitOp>(operation)) {
    serializeQuregSplit(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::QuregJoinOp>(operation)) {
    serializeQuregJoin(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::QuregCreateOp>(operation)) {
    serializeQuregCreate(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::QuregFreeOp>(operation)) {
    serializeQuregFree(builder, op, ctx);
  } else {
    llvm::errs() << "Cannot serialize qureg operation " << operation->getName()
                 << "\n";
    llvm::report_fatal_error("Unknown qureg operation");
  }
}

//===----------------------------------------------------------------------===//
// Int operations
//===----------------------------------------------------------------------===//

#define SERIALIZE_INT_CONST(BIT_WIDTH)                                         \
  void serializeIntConst##BIT_WIDTH(jeff::Op::Builder builder,                 \
                                    mlir::jeff::IntConst##BIT_WIDTH##Op op,    \
                                    SerializationContext& ctx) {               \
    auto intBuilder = builder.initInstruction().initInt();                     \
    intBuilder.setConst##BIT_WIDTH(op.getVal());                               \
                                                                               \
    builder.initInputs(0);                                                     \
                                                                               \
    auto outputs = builder.initOutputs(1);                                     \
    outputs.set(0, ctx.getValueId(op.getConstant()));                          \
  }

SERIALIZE_INT_CONST(1)
SERIALIZE_INT_CONST(8)
SERIALIZE_INT_CONST(16)
SERIALIZE_INT_CONST(32)
SERIALIZE_INT_CONST(64)

#undef SERIALIZE_INT_CONST

void serializeIntUnary(jeff::Op::Builder builder, mlir::jeff::IntUnaryOp op,
                       SerializationContext& ctx) {
  auto intBuilder = builder.initInstruction().initInt();
  switch (op.getOp()) {
  case mlir::jeff::IntUnaryOperation::_not:
    intBuilder.setNot();
    break;
  case mlir::jeff::IntUnaryOperation::_abs:
    intBuilder.setAbs();
    break;
  default:
    llvm::report_fatal_error("Unknown int unary operation");
  }

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getA()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getB()));
}

void serializeIntBinary(jeff::Op::Builder builder, mlir::jeff::IntBinaryOp op,
                        SerializationContext& ctx) {
  auto intBuilder = builder.initInstruction().initInt();
  switch (op.getOp()) {
  case mlir::jeff::IntBinaryOperation::_add:
    intBuilder.setAdd();
    break;
  case mlir::jeff::IntBinaryOperation::_sub:
    intBuilder.setSub();
    break;
  case mlir::jeff::IntBinaryOperation::_mul:
    intBuilder.setMul();
    break;
  case mlir::jeff::IntBinaryOperation::_divS:
    intBuilder.setDivS();
    break;
  case mlir::jeff::IntBinaryOperation::_divU:
    intBuilder.setDivU();
    break;
  case mlir::jeff::IntBinaryOperation::_pow:
    intBuilder.setPow();
    break;
  case mlir::jeff::IntBinaryOperation::_and:
    intBuilder.setAnd();
    break;
  case mlir::jeff::IntBinaryOperation::_or:
    intBuilder.setOr();
    break;
  case mlir::jeff::IntBinaryOperation::_xor:
    intBuilder.setXor();
    break;
  case mlir::jeff::IntBinaryOperation::_minS:
    intBuilder.setMinS();
    break;
  case mlir::jeff::IntBinaryOperation::_minU:
    intBuilder.setMinU();
    break;
  case mlir::jeff::IntBinaryOperation::_maxS:
    intBuilder.setMaxS();
    break;
  case mlir::jeff::IntBinaryOperation::_maxU:
    intBuilder.setMaxU();
    break;
  case mlir::jeff::IntBinaryOperation::_remS:
    intBuilder.setRemS();
    break;
  case mlir::jeff::IntBinaryOperation::_remU:
    intBuilder.setRemU();
    break;
  case mlir::jeff::IntBinaryOperation::_shl:
    intBuilder.setShl();
    break;
  case mlir::jeff::IntBinaryOperation::_shr:
    intBuilder.setShr();
    break;
  default:
    llvm::report_fatal_error("Unknown int binary operation");
  }

  auto inputs = builder.initInputs(2);
  inputs.set(0, ctx.getValueId(op.getA()));
  inputs.set(1, ctx.getValueId(op.getB()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getC()));
}

void serializeIntComparison(jeff::Op::Builder builder,
                            mlir::jeff::IntComparisonOp op,
                            SerializationContext& ctx) {
  auto intBuilder = builder.initInstruction().initInt();
  switch (op.getOp()) {
  case mlir::jeff::IntComparisonOperation::_eq:
    intBuilder.setEq();
    break;
  case mlir::jeff::IntComparisonOperation::_ltS:
    intBuilder.setLtS();
    break;
  case mlir::jeff::IntComparisonOperation::_lteS:
    intBuilder.setLteS();
    break;
  case mlir::jeff::IntComparisonOperation::_ltU:
    intBuilder.setLtU();
    break;
  case mlir::jeff::IntComparisonOperation::_lteU:
    intBuilder.setLteU();
    break;
  default:
    llvm::report_fatal_error("Unknown int comparison operation");
  }

  auto inputs = builder.initInputs(2);
  inputs.set(0, ctx.getValueId(op.getA()));
  inputs.set(1, ctx.getValueId(op.getB()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getResult()));
}

void serializeInt(jeff::Op::Builder builder, mlir::Operation* operation,
                  SerializationContext& ctx) {
  if (auto op = llvm::dyn_cast<mlir::jeff::IntConst1Op>(operation)) {
    serializeIntConst1(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::IntConst8Op>(operation)) {
    serializeIntConst8(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::IntConst16Op>(operation)) {
    serializeIntConst16(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::IntConst32Op>(operation)) {
    serializeIntConst32(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::IntConst64Op>(operation)) {
    serializeIntConst64(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::IntUnaryOp>(operation)) {
    serializeIntUnary(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::IntBinaryOp>(operation)) {
    serializeIntBinary(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::IntComparisonOp>(operation)) {
    serializeIntComparison(builder, op, ctx);
  } else {
    llvm::errs() << "Cannot serialize int operation " << operation->getName()
                 << "\n";
    llvm::report_fatal_error("Unknown int operation");
  }
}

//===----------------------------------------------------------------------===//
// IntArray operations
//===----------------------------------------------------------------------===//

void serializeIntArrayConst1(jeff::Op::Builder builder,
                             mlir::jeff::IntArrayConst1Op op,
                             SerializationContext& ctx) {
  auto intArrayBuilder = builder.initInstruction().initIntArray();
  auto values = op.getInArray();
  auto listBuilder = intArrayBuilder.initConst1(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    listBuilder.set(i, static_cast<bool>(values[i]));
  }

  builder.initInputs(0);

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutArray()));
}

#define SERIALIZE_INT_ARRAY_CONST(BIT_WIDTH)                                   \
  void serializeIntArrayConst##BIT_WIDTH(                                      \
      jeff::Op::Builder builder, mlir::jeff::IntArrayConst##BIT_WIDTH##Op op,  \
      SerializationContext& ctx) {                                             \
    auto intArrayBuilder = builder.initInstruction().initIntArray();           \
    auto values = op.getInArray();                                             \
    auto listBuilder = intArrayBuilder.initConst##BIT_WIDTH(values.size());    \
    for (size_t i = 0; i < values.size(); ++i) {                               \
      listBuilder.set(i, static_cast<uint##BIT_WIDTH##_t>(values[i]));         \
    }                                                                          \
                                                                               \
    builder.initInputs(0);                                                     \
                                                                               \
    auto outputs = builder.initOutputs(1);                                     \
    outputs.set(0, ctx.getValueId(op.getOutArray()));                          \
  }

SERIALIZE_INT_ARRAY_CONST(8)
SERIALIZE_INT_ARRAY_CONST(16)
SERIALIZE_INT_ARRAY_CONST(32)
SERIALIZE_INT_ARRAY_CONST(64)

#undef SERIALIZE_INT_ARRAY_CONST

void serializeIntArrayZero(jeff::Op::Builder builder,
                           mlir::jeff::IntArrayZeroOp op,
                           SerializationContext& ctx) {
  auto intArrayBuilder = builder.initInstruction().initIntArray();
  auto tensorType =
      llvm::cast<mlir::RankedTensorType>(op.getOutArray().getType());
  auto elementType = llvm::cast<mlir::IntegerType>(tensorType.getElementType());
  intArrayBuilder.setZero(elementType.getWidth());

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getLength()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeIntArrayGetIndex(jeff::Op::Builder builder,
                               mlir::jeff::IntArrayGetIndexOp op,
                               SerializationContext& ctx) {
  auto intArrayBuilder = builder.initInstruction().initIntArray();
  intArrayBuilder.setGetIndex();

  auto inputs = builder.initInputs(2);
  inputs.set(0, ctx.getValueId(op.getInArray()));
  inputs.set(1, ctx.getValueId(op.getIndex()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getValue()));
}

void serializeIntArraySetIndex(jeff::Op::Builder builder,
                               mlir::jeff::IntArraySetIndexOp op,
                               SerializationContext& ctx) {
  auto intArrayBuilder = builder.initInstruction().initIntArray();
  intArrayBuilder.setSetIndex();

  auto inputs = builder.initInputs(3);
  inputs.set(0, ctx.getValueId(op.getInArray()));
  inputs.set(1, ctx.getValueId(op.getIndex()));
  inputs.set(2, ctx.getValueId(op.getValue()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeIntArrayLength(jeff::Op::Builder builder,
                             mlir::jeff::IntArrayLengthOp op,
                             SerializationContext& ctx) {
  auto intArrayBuilder = builder.initInstruction().initIntArray();
  intArrayBuilder.setLength();

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getInArray()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getLength()));
}

void serializeIntArrayCreate(jeff::Op::Builder builder,
                             mlir::jeff::IntArrayCreateOp op,
                             SerializationContext& ctx) {
  auto intArrayBuilder = builder.initInstruction().initIntArray();
  intArrayBuilder.setCreate();

  auto inputs = builder.initInputs(op.getInArray().size());
  for (size_t i = 0; i < op.getInArray().size(); ++i) {
    inputs.set(i, ctx.getValueId(op.getInArray()[i]));
  }

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeIntArray(jeff::Op::Builder builder, mlir::Operation* operation,
                       SerializationContext& ctx) {
  if (auto op = llvm::dyn_cast<mlir::jeff::IntArrayConst1Op>(operation)) {
    serializeIntArrayConst1(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::IntArrayConst8Op>(operation)) {
    serializeIntArrayConst8(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::IntArrayConst16Op>(operation)) {
    serializeIntArrayConst16(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::IntArrayConst32Op>(operation)) {
    serializeIntArrayConst32(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::IntArrayConst64Op>(operation)) {
    serializeIntArrayConst64(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::IntArrayZeroOp>(operation)) {
    serializeIntArrayZero(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::IntArrayGetIndexOp>(operation)) {
    serializeIntArrayGetIndex(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::IntArraySetIndexOp>(operation)) {
    serializeIntArraySetIndex(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::IntArrayLengthOp>(operation)) {
    serializeIntArrayLength(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::IntArrayCreateOp>(operation)) {
    serializeIntArrayCreate(builder, op, ctx);
  } else {
    llvm::errs() << "Cannot serialize int array operation "
                 << operation->getName() << "\n";
    llvm::report_fatal_error("Unknown int array operation");
  }
}

//===----------------------------------------------------------------------===//
// Float operations
//===----------------------------------------------------------------------===//

void serializeFloatConst32(jeff::Op::Builder builder,
                           mlir::jeff::FloatConst32Op op,
                           SerializationContext& ctx) {
  auto floatBuilder = builder.initInstruction().initFloat();
  auto value = static_cast<float>(op.getVal().convertToDouble());
  floatBuilder.setConst32(value);

  builder.initInputs(0);

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getConstant()));
}

void serializeFloatConst64(jeff::Op::Builder builder,
                           mlir::jeff::FloatConst64Op op,
                           SerializationContext& ctx) {
  auto floatBuilder = builder.initInstruction().initFloat();
  auto value = op.getVal().convertToDouble();
  floatBuilder.setConst64(value);

  builder.initInputs(0);

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getConstant()));
}

void serializeFloatUnary(jeff::Op::Builder builder, mlir::jeff::FloatUnaryOp op,
                         SerializationContext& ctx) {
  auto floatBuilder = builder.initInstruction().initFloat();
  switch (op.getOp()) {
  case mlir::jeff::FloatUnaryOperation::_sqrt:
    floatBuilder.setSqrt();
    break;
  case mlir::jeff::FloatUnaryOperation::_abs:
    floatBuilder.setAbs();
    break;
  case mlir::jeff::FloatUnaryOperation::_ceil:
    floatBuilder.setCeil();
    break;
  case mlir::jeff::FloatUnaryOperation::_floor:
    floatBuilder.setFloor();
    break;
  case mlir::jeff::FloatUnaryOperation::_exp:
    floatBuilder.setExp();
    break;
  case mlir::jeff::FloatUnaryOperation::_log:
    floatBuilder.setLog();
    break;
  case mlir::jeff::FloatUnaryOperation::_sin:
    floatBuilder.setSin();
    break;
  case mlir::jeff::FloatUnaryOperation::_cos:
    floatBuilder.setCos();
    break;
  case mlir::jeff::FloatUnaryOperation::_tan:
    floatBuilder.setTan();
    break;
  case mlir::jeff::FloatUnaryOperation::_asin:
    floatBuilder.setAsin();
    break;
  case mlir::jeff::FloatUnaryOperation::_acos:
    floatBuilder.setAcos();
    break;
  case mlir::jeff::FloatUnaryOperation::_atan:
    floatBuilder.setAtan();
    break;
  case mlir::jeff::FloatUnaryOperation::_sinh:
    floatBuilder.setSinh();
    break;
  case mlir::jeff::FloatUnaryOperation::_cosh:
    floatBuilder.setCosh();
    break;
  case mlir::jeff::FloatUnaryOperation::_tanh:
    floatBuilder.setTanh();
    break;
  case mlir::jeff::FloatUnaryOperation::_asinh:
    floatBuilder.setAsinh();
    break;
  case mlir::jeff::FloatUnaryOperation::_acosh:
    floatBuilder.setAcosh();
    break;
  case mlir::jeff::FloatUnaryOperation::_atanh:
    floatBuilder.setAtanh();
    break;
  default:
    llvm::report_fatal_error("Unknown float unary operation");
  }

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getA()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getB()));
}

void serializeFloatBinary(jeff::Op::Builder builder,
                          mlir::jeff::FloatBinaryOp op,
                          SerializationContext& ctx) {
  auto floatBuilder = builder.initInstruction().initFloat();
  switch (op.getOp()) {
  case mlir::jeff::FloatBinaryOperation::_add:
    floatBuilder.setAdd();
    break;
  case mlir::jeff::FloatBinaryOperation::_sub:
    floatBuilder.setSub();
    break;
  case mlir::jeff::FloatBinaryOperation::_mul:
    floatBuilder.setMul();
    break;
  case mlir::jeff::FloatBinaryOperation::_pow:
    floatBuilder.setPow();
    break;
  case mlir::jeff::FloatBinaryOperation::_atan2:
    floatBuilder.setAtan2();
    break;
  case mlir::jeff::FloatBinaryOperation::_max:
    floatBuilder.setMax();
    break;
  case mlir::jeff::FloatBinaryOperation::_min:
    floatBuilder.setMin();
    break;
  default:
    llvm::report_fatal_error("Unknown float binary operation");
  }

  auto inputs = builder.initInputs(2);
  inputs.set(0, ctx.getValueId(op.getA()));
  inputs.set(1, ctx.getValueId(op.getB()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getC()));
}

void serializeFloatComparison(jeff::Op::Builder builder,
                              mlir::jeff::FloatComparisonOp op,
                              SerializationContext& ctx) {
  auto floatBuilder = builder.initInstruction().initFloat();
  switch (op.getOp()) {
  case mlir::jeff::FloatComparisonOperation::_eq:
    floatBuilder.setEq();
    break;
  case mlir::jeff::FloatComparisonOperation::_lt:
    floatBuilder.setLt();
    break;
  case mlir::jeff::FloatComparisonOperation::_lte:
    floatBuilder.setLte();
    break;
  default:
    llvm::report_fatal_error("Unknown float comparison operation");
  }

  auto inputs = builder.initInputs(2);
  inputs.set(0, ctx.getValueId(op.getA()));
  inputs.set(1, ctx.getValueId(op.getB()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getC()));
}

void serializeFloatIs(jeff::Op::Builder builder, mlir::jeff::FloatIsOp op,
                      SerializationContext& ctx) {
  auto floatBuilder = builder.initInstruction().initFloat();
  switch (op.getOp()) {
  case mlir::jeff::FloatIsOperation::_isNan:
    floatBuilder.setIsNan();
    break;
  case mlir::jeff::FloatIsOperation::_isInf:
    floatBuilder.setIsInf();
    break;
  default:
    llvm::report_fatal_error("Unknown float is operation");
  }

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getA()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getResult()));
}

void serializeFloat(jeff::Op::Builder builder, mlir::Operation* operation,
                    SerializationContext& ctx) {
  if (auto op = llvm::dyn_cast<mlir::jeff::FloatConst32Op>(operation)) {
    serializeFloatConst32(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::FloatConst64Op>(operation)) {
    serializeFloatConst64(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::FloatUnaryOp>(operation)) {
    serializeFloatUnary(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::FloatBinaryOp>(operation)) {
    serializeFloatBinary(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::FloatComparisonOp>(operation)) {
    serializeFloatComparison(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::FloatIsOp>(operation)) {
    serializeFloatIs(builder, op, ctx);
  } else {
    llvm::errs() << "Cannot serialize float operation " << operation->getName()
                 << "\n";
    llvm::report_fatal_error("Unknown float operation");
  }
}

//===----------------------------------------------------------------------===//
// FloatArray operations
//===----------------------------------------------------------------------===//

void serializeFloatArrayConst32(jeff::Op::Builder builder,
                                mlir::jeff::FloatArrayConst32Op op,
                                SerializationContext& ctx) {
  auto floatArrayBuilder = builder.initInstruction().initFloatArray();
  auto values = op.getInArray();
  auto listBuilder = floatArrayBuilder.initConst32(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    listBuilder.set(i, static_cast<float>(values[i]));
  }

  builder.initInputs(0);

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeFloatArrayConst64(jeff::Op::Builder builder,
                                mlir::jeff::FloatArrayConst64Op op,
                                SerializationContext& ctx) {
  auto floatArrayBuilder = builder.initInstruction().initFloatArray();
  auto values = op.getInArray();
  auto listBuilder = floatArrayBuilder.initConst64(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    listBuilder.set(i, static_cast<double>(values[i]));
  }

  builder.initInputs(0);

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeFloatArrayZero(jeff::Op::Builder builder,
                             mlir::jeff::FloatArrayZeroOp op,
                             SerializationContext& ctx) {
  auto floatArrayBuilder = builder.initInstruction().initFloatArray();
  auto tensorType =
      llvm::cast<mlir::RankedTensorType>(op.getOutArray().getType());
  auto elementType = llvm::cast<mlir::FloatType>(tensorType.getElementType());
  if (elementType.getWidth() == 32) {
    floatArrayBuilder.setZero(jeff::FloatPrecision::FLOAT32);
  } else if (elementType.getWidth() == 64) {
    floatArrayBuilder.setZero(jeff::FloatPrecision::FLOAT64);
  } else {
    llvm::report_fatal_error("Unknown float array element type");
  }

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getLength()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeFloatArrayGetIndex(jeff::Op::Builder builder,
                                 mlir::jeff::FloatArrayGetIndexOp op,
                                 SerializationContext& ctx) {
  auto floatArrayBuilder = builder.initInstruction().initFloatArray();
  floatArrayBuilder.setGetIndex();

  auto inputs = builder.initInputs(2);
  inputs.set(0, ctx.getValueId(op.getInArray()));
  inputs.set(1, ctx.getValueId(op.getIndex()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getValue()));
}

void serializeFloatArraySetIndex(jeff::Op::Builder builder,
                                 mlir::jeff::FloatArraySetIndexOp op,
                                 SerializationContext& ctx) {
  auto floatArrayBuilder = builder.initInstruction().initFloatArray();
  floatArrayBuilder.setSetIndex();

  auto inputs = builder.initInputs(3);
  inputs.set(0, ctx.getValueId(op.getInArray()));
  inputs.set(1, ctx.getValueId(op.getIndex()));
  inputs.set(2, ctx.getValueId(op.getValue()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeFloatArrayLength(jeff::Op::Builder builder,
                               mlir::jeff::FloatArrayLengthOp op,
                               SerializationContext& ctx) {
  auto floatArrayBuilder = builder.initInstruction().initFloatArray();
  floatArrayBuilder.setLength();

  auto inputs = builder.initInputs(1);
  inputs.set(0, ctx.getValueId(op.getInArray()));

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getLength()));
}

void serializeFloatArrayCreate(jeff::Op::Builder builder,
                               mlir::jeff::FloatArrayCreateOp op,
                               SerializationContext& ctx) {
  auto floatArrayBuilder = builder.initInstruction().initFloatArray();
  floatArrayBuilder.setCreate();

  auto inputs = builder.initInputs(op.getInArray().size());
  for (size_t i = 0; i < op.getInArray().size(); ++i) {
    inputs.set(i, ctx.getValueId(op.getInArray()[i]));
  }

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getOutArray()));
}

void serializeFloatArray(jeff::Op::Builder builder, mlir::Operation* operation,
                         SerializationContext& ctx) {
  if (auto op = llvm::dyn_cast<mlir::jeff::FloatArrayConst32Op>(operation)) {
    serializeFloatArrayConst32(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::FloatArrayConst64Op>(operation)) {
    serializeFloatArrayConst64(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::FloatArrayZeroOp>(operation)) {
    serializeFloatArrayZero(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::FloatArrayGetIndexOp>(operation)) {
    serializeFloatArrayGetIndex(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::FloatArraySetIndexOp>(operation)) {
    serializeFloatArraySetIndex(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::FloatArrayLengthOp>(operation)) {
    serializeFloatArrayLength(builder, op, ctx);
  } else if (auto op =
                 llvm::dyn_cast<mlir::jeff::FloatArrayCreateOp>(operation)) {
    serializeFloatArrayCreate(builder, op, ctx);
  } else {
    llvm::errs() << "Cannot serialize float array operation "
                 << operation->getName() << "\n";
    llvm::report_fatal_error("Unknown float array operation");
  }
}
//===----------------------------------------------------------------------===//
// SCF operations
//===----------------------------------------------------------------------===//

// Forward declaration
void serializeOperation(jeff::Op::Builder builder, mlir::Operation* operation,
                        SerializationContext& ctx);

void serializeSwitch(jeff::Op::Builder builder, mlir::jeff::SwitchOp op,
                     SerializationContext& ctx) {
  auto switchBuilder = builder.initInstruction().initScf().initSwitch();

  const auto numInputs = op.getNumOperands();
  auto inputs = builder.initInputs(numInputs);
  for (size_t i = 0; i < numInputs; ++i) {
    inputs.set(i, ctx.getValueId(op.getOperand(i)));
  }

  const auto numOutputs = op.getNumResults();
  auto outputs = builder.initOutputs(numOutputs);
  for (size_t i = 0; i < numOutputs; ++i) {
    outputs.set(i, ctx.getValueId(op.getResult(i)));
  }

  auto branches = op.getBranches();
  const auto numBranches = branches.size();
  auto branchBuilders = switchBuilder.initBranches(numBranches);
  for (size_t i = 0; i < numBranches; ++i) {
    auto& block = branches[i].front();
    auto branchBuilder = branchBuilders[i];

    const auto numSources = block.getNumArguments();
    auto sources = branchBuilder.initSources(numSources);
    for (size_t j = 0; j < numSources; ++j) {
      sources.set(j, ctx.getValueId(block.getArgument(j)));
    }

    const auto numOperations = block.getOperations().size() - 1;
    auto operationBuilders = branchBuilder.initOperations(numOperations);
    size_t j = 0;
    for (auto& operation : block.getOperations()) {
      if (llvm::isa<mlir::jeff::YieldOp>(operation)) {
        continue;
      }
      serializeOperation(operationBuilders[j], &operation, ctx);
      ++j;
    }

    auto yieldOp = llvm::cast<mlir::jeff::YieldOp>(block.back());
    const auto numTargets = yieldOp.getNumOperands();
    auto targets = branchBuilder.initTargets(numTargets);
    for (size_t j = 0; j < numTargets; ++j) {
      targets.set(j, ctx.getValueId(yieldOp.getOperand(j)));
    }
  }

  {
    auto& block = op.getDefault().front();
    auto defaultBuilder = switchBuilder.initDefault();

    const auto numSources = block.getNumArguments();
    auto sources = defaultBuilder.initSources(numSources);
    for (size_t i = 0; i < numSources; ++i) {
      sources.set(i, ctx.getValueId(block.getArgument(i)));
    }

    const auto numOperations = block.getOperations().size() - 1;
    auto operationBuilders = defaultBuilder.initOperations(numOperations);
    size_t i = 0;
    for (auto& operation : block.getOperations()) {
      if (llvm::isa<mlir::jeff::YieldOp>(operation)) {
        continue;
      }
      serializeOperation(operationBuilders[i], &operation, ctx);
      ++i;
    }

    auto yieldOp = llvm::cast<mlir::jeff::YieldOp>(block.back());
    const auto numTargets = yieldOp.getNumOperands();
    auto targets = defaultBuilder.initTargets(numTargets);
    for (size_t i = 0; i < numTargets; ++i) {
      targets.set(i, ctx.getValueId(yieldOp.getOperand(i)));
    }
  }
}

void serializeFor(jeff::Op::Builder builder, mlir::jeff::ForOp op,
                  SerializationContext& ctx) {
  auto forBuilder = builder.initInstruction().initScf().initFor();

  const auto numInputs = op.getNumOperands();
  auto inputs = builder.initInputs(numInputs);
  for (size_t i = 0; i < numInputs; ++i) {
    inputs.set(i, ctx.getValueId(op.getOperand(i)));
  }

  const auto numOutputs = op.getNumResults();
  auto outputs = builder.initOutputs(numOutputs);
  for (size_t i = 0; i < numOutputs; ++i) {
    outputs.set(i, ctx.getValueId(op.getResult(i)));
  }

  auto& block = op.getBody().front();

  const auto numSources = block.getNumArguments();
  auto sources = forBuilder.initSources(numSources);
  for (size_t i = 0; i < numSources; ++i) {
    sources.set(i, ctx.getValueId(block.getArgument(i)));
  }

  const auto numOperations = block.getOperations().size() - 1;
  auto operationBuilders = forBuilder.initOperations(numOperations);
  size_t i = 0;
  for (auto& operation : block.getOperations()) {
    if (llvm::isa<mlir::jeff::YieldOp>(operation)) {
      continue;
    }
    serializeOperation(operationBuilders[i], &operation, ctx);
    ++i;
  }

  auto yieldOp = llvm::cast<mlir::jeff::YieldOp>(block.back());
  const auto numTargets = yieldOp.getNumOperands();
  auto targets = forBuilder.initTargets(numTargets);
  for (size_t i = 0; i < numTargets; ++i) {
    targets.set(i, ctx.getValueId(yieldOp.getOperand(i)));
  }
}

void serializeWhile(jeff::Op::Builder builder, mlir::jeff::WhileOp op,
                    SerializationContext& ctx) {
  auto whileBuilder = builder.initInstruction().initScf().initWhile();

  const auto numInputs = op.getNumOperands();
  auto inputs = builder.initInputs(numInputs);
  for (size_t i = 0; i < numInputs; ++i) {
    inputs.set(i, ctx.getValueId(op.getOperand(i)));
  }

  const auto numOutputs = op.getNumResults();
  auto outputs = builder.initOutputs(numOutputs);
  for (size_t i = 0; i < numOutputs; ++i) {
    outputs.set(i, ctx.getValueId(op.getResult(i)));
  }

  {
    auto& condition = op.getCondition();
    auto conditionBuilder = whileBuilder.initCondition();

    const auto numSources = condition.getNumArguments();
    auto sources = conditionBuilder.initSources(numSources);
    for (size_t i = 0; i < numSources; ++i) {
      sources.set(i, ctx.getValueId(condition.getArgument(i)));
    }

    const auto numOperations = condition.front().getOperations().size() - 1;
    auto operationBuilders = conditionBuilder.initOperations(numOperations);
    size_t i = 0;
    for (auto& operation : condition.front().getOperations()) {
      if (llvm::isa<mlir::jeff::YieldOp>(operation)) {
        continue;
      }
      serializeOperation(operationBuilders[i], &operation, ctx);
      ++i;
    }

    auto yieldOp = llvm::cast<mlir::jeff::YieldOp>(condition.front().back());
    auto targets = conditionBuilder.initTargets(1);
    targets.set(0, ctx.getValueId(yieldOp.getOperand(0)));
  }

  {
    auto& body = op.getBody();
    auto bodyBuilder = whileBuilder.initBody();

    const auto numSources = body.getNumArguments();
    auto sources = bodyBuilder.initSources(numSources);
    for (size_t i = 0; i < numSources; ++i) {
      sources.set(i, ctx.getValueId(body.getArgument(i)));
    }

    const auto numOperations = body.front().getOperations().size() - 1;
    auto operationBuilders = bodyBuilder.initOperations(numOperations);
    size_t i = 0;
    for (auto& operation : body.front().getOperations()) {
      if (llvm::isa<mlir::jeff::YieldOp>(operation)) {
        continue;
      }
      serializeOperation(operationBuilders[i], &operation, ctx);
      ++i;
    }

    auto yieldOp = llvm::cast<mlir::jeff::YieldOp>(body.front().back());
    const auto numTargets = yieldOp.getNumOperands();
    auto targets = bodyBuilder.initTargets(numTargets);
    for (size_t i = 0; i < numTargets; ++i) {
      targets.set(i, ctx.getValueId(yieldOp.getOperand(i)));
    }
  }
}

void serializeDoWhile(jeff::Op::Builder builder, mlir::jeff::DoWhileOp op,
                      SerializationContext& ctx) {
  auto doWhileBuilder = builder.initInstruction().initScf().initDoWhile();

  const auto numInputs = op.getNumOperands();
  auto inputs = builder.initInputs(numInputs);
  for (size_t i = 0; i < numInputs; ++i) {
    inputs.set(i, ctx.getValueId(op.getOperand(i)));
  }

  const auto numOutputs = op.getNumResults();
  auto outputs = builder.initOutputs(numOutputs);
  for (size_t i = 0; i < numOutputs; ++i) {
    outputs.set(i, ctx.getValueId(op.getResult(i)));
  }

  {
    auto& condition = op.getCondition();
    auto conditionBuilder = doWhileBuilder.initCondition();

    const auto numSources = condition.getNumArguments();
    auto sources = conditionBuilder.initSources(numSources);
    for (size_t i = 0; i < numSources; ++i) {
      sources.set(i, ctx.getValueId(condition.getArgument(i)));
    }

    const auto numOperations = condition.front().getOperations().size() - 1;
    auto operationBuilders = conditionBuilder.initOperations(numOperations);
    size_t i = 0;
    for (auto& operation : condition.front().getOperations()) {
      if (llvm::isa<mlir::jeff::YieldOp>(operation)) {
        continue;
      }
      serializeOperation(operationBuilders[i], &operation, ctx);
      ++i;
    }

    auto yieldOp = llvm::cast<mlir::jeff::YieldOp>(condition.front().back());
    auto targets = conditionBuilder.initTargets(1);
    targets.set(0, ctx.getValueId(yieldOp.getOperand(0)));
  }

  {
    auto& body = op.getBody();
    auto bodyBuilder = doWhileBuilder.initBody();

    const auto numSources = body.getNumArguments();
    auto sources = bodyBuilder.initSources(numSources);
    for (size_t i = 0; i < numSources; ++i) {
      sources.set(i, ctx.getValueId(body.getArgument(i)));
    }

    const auto numOperations = body.front().getOperations().size() - 1;
    auto operationBuilders = bodyBuilder.initOperations(numOperations);
    size_t i = 0;
    for (auto& operation : body.front().getOperations()) {
      if (llvm::isa<mlir::jeff::YieldOp>(operation)) {
        continue;
      }
      serializeOperation(operationBuilders[i], &operation, ctx);
      ++i;
    }

    auto yieldOp = llvm::cast<mlir::jeff::YieldOp>(body.front().back());
    const auto numTargets = yieldOp.getNumOperands();
    auto targets = bodyBuilder.initTargets(numTargets);
    for (size_t i = 0; i < numTargets; ++i) {
      targets.set(i, ctx.getValueId(yieldOp.getOperand(i)));
    }
  }
}

void serializeSCF(jeff::Op::Builder builder, mlir::Operation* operation,
                  SerializationContext& ctx) {
  if (auto op = llvm::dyn_cast<mlir::jeff::SwitchOp>(operation)) {
    serializeSwitch(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::ForOp>(operation)) {
    serializeFor(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::WhileOp>(operation)) {
    serializeWhile(builder, op, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::jeff::DoWhileOp>(operation)) {
    serializeDoWhile(builder, op, ctx);
  } else {
    llvm::errs() << "Cannot serialize SCF operation " << operation->getName()
                 << "\n";
    llvm::report_fatal_error("Unknown SCF operation");
  }
}

//===----------------------------------------------------------------------===//
// Func operations
//===----------------------------------------------------------------------===//

void serializeCall(jeff::Op::Builder builder, mlir::func::CallOp op,
                   SerializationContext& ctx) {
  auto funcBuilder = builder.initInstruction().initFunc();
  funcBuilder.setFuncCall(ctx.getFuncId(op.getCallee()));

  const auto numInputs = op.getNumOperands();
  auto inputs = builder.initInputs(numInputs);
  for (size_t i = 0; i < numInputs; ++i) {
    inputs.set(i, ctx.getValueId(op.getOperand(i)));
  }

  const auto numOutputs = op.getNumResults();
  auto outputs = builder.initOutputs(numOutputs);
  for (size_t i = 0; i < numOutputs; ++i) {
    outputs.set(i, ctx.getValueId(op.getResult(i)));
  }
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

void serializeQubitType(jeff::Type::Builder builder) { builder.setQubit(); }

void serializeQuregType(jeff::Type::Builder builder) { builder.setQureg(); }

void serializeIntType(jeff::Type::Builder builder, mlir::Type type) {
  auto intType = llvm::cast<mlir::IntegerType>(type);
  builder.setInt(intType.getWidth());
}

void serializeIntArrayType(jeff::Type::Builder builder, mlir::Type type) {
  auto tensorType = llvm::cast<mlir::RankedTensorType>(type);
  auto elementType = llvm::cast<mlir::IntegerType>(tensorType.getElementType());
  builder.setIntArray(elementType.getWidth());
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

void serializeOperation(jeff::Op::Builder builder, mlir::Operation* operation,
                        SerializationContext& ctx) {
  if (llvm::isa<mlir::jeff::QubitOperation>(operation)) {
    serializeQubit(builder, operation, ctx);
  } else if (llvm::isa<mlir::jeff::QuregOperation>(operation)) {
    serializeQureg(builder, operation, ctx);
  } else if (llvm::isa<mlir::jeff::IntOperation>(operation)) {
    serializeInt(builder, operation, ctx);
  } else if (llvm::isa<mlir::jeff::IntArrayOperation>(operation)) {
    serializeIntArray(builder, operation, ctx);
  } else if (llvm::isa<mlir::jeff::FloatOperation>(operation)) {
    serializeFloat(builder, operation, ctx);
  } else if (llvm::isa<mlir::jeff::FloatArrayOperation>(operation)) {
    serializeFloatArray(builder, operation, ctx);
  } else if (llvm::isa<mlir::jeff::SCFOperation>(operation)) {
    serializeSCF(builder, operation, ctx);
  } else if (auto op = llvm::dyn_cast<mlir::func::CallOp>(operation)) {
    serializeCall(builder, op, ctx);
  } else {
    llvm::errs() << "Cannot serialize operation " << operation->getName()
                 << "\n";
    llvm::report_fatal_error("Unknown operation");
  }
}

void serializeFunction(jeff::Function::Builder funcBuilder,
                       mlir::func::FuncOp func, SerializationContext& ctx) {
  ctx.values.clear();

  auto defBuilder = funcBuilder.initDefinition();
  auto& entryBlock = func.getRegion().front();

  // Build body
  auto bodyBuilder = defBuilder.initBody();

  // Set sources
  const auto numSources = entryBlock.getNumArguments();
  auto sourcesBuilder = bodyBuilder.initSources(numSources);
  for (unsigned i = 0; i < numSources; ++i) {
    sourcesBuilder.set(i, ctx.getValueId(entryBlock.getArgument(i)));
  }

  // Serialize operations
  const auto numOperations = entryBlock.getOperations().size() - 1;
  auto operationBuilders = bodyBuilder.initOperations(numOperations);
  size_t i = 0;
  for (auto& operation : entryBlock.getOperations()) {
    if (llvm::isa<mlir::func::ReturnOp>(operation)) {
      continue;
    }
    serializeOperation(operationBuilders[i], &operation, ctx);
    ++i;
  }

  // Set targets
  auto returnOp = llvm::cast<mlir::func::ReturnOp>(entryBlock.back());
  const auto numTargets = returnOp.getNumOperands();
  auto targetsBuilder = bodyBuilder.initTargets(numTargets);
  for (unsigned i = 0; i < numTargets; ++i) {
    targetsBuilder.set(i, ctx.getValueId(returnOp.getOperand(i)));
  }

  // Build values
  const auto numValues = ctx.values.size();
  auto valuesBuilder = defBuilder.initValues(numValues);
  llvm::SmallVector<mlir::Value> values(numValues);
  for (auto& pair : ctx.values) {
    values[pair.second] = pair.first;
  }
  for (size_t i = 0, j = 0; i < numValues; ++i) {
    auto valueBuilder = valuesBuilder[i];
    auto typeBuilder = valueBuilder.initType();
    serializeType(typeBuilder, values[i].getType());
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
  const auto numStrings = stringsAttr.size();
  auto stringsBuilder = moduleBuilder.initStrings(numStrings);
  for (auto i = 0; i < numStrings; ++i) {
    const auto str =
        llvm::cast<mlir::StringAttr>(stringsAttr[i]).getValue().str();
    ctx.strings[str] = i;
    stringsBuilder.set(i, str);
  }

  // Build functions
  uint32_t id = 0;
  llvm::SmallVector<mlir::func::FuncOp> functions;
  module.walk([&](mlir::func::FuncOp func) {
    ctx.funcs[func.getSymName()] = id++;
    functions.push_back(func);
  });

  const auto numFunctions = functions.size();
  auto functionBuilders = moduleBuilder.initFunctions(numFunctions);

  for (size_t i = 0; i < numFunctions; ++i) {
    auto function = functions[i];
    auto functionBuilder = functionBuilders[i];
    functionBuilder.setName(ctx.getStringId(function.getName().str()));
    serializeFunction(functionBuilder, function, ctx);
  }

  // Set metadata
  moduleBuilder.setEntrypoint(
      llvm::cast<mlir::IntegerAttr>(module->getAttr("jeff.entrypoint"))
          .getInt());

  moduleBuilder.setTool(
      llvm::cast<mlir::StringAttr>(module->getAttr("jeff.tool"))
          .getValue()
          .str());

  moduleBuilder.setToolVersion(
      llvm::cast<mlir::StringAttr>(module->getAttr("jeff.toolVersion"))
          .getValue()
          .str());

  return capnp::messageToFlatArray(message);
}
