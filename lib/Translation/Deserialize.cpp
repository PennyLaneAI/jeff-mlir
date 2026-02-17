#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffOps.h"

#include <capnp/common.h>
#include <capnp/list.h>
#include <capnp/serialize.h>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <jeff.capnp.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <string>

namespace {

struct DeserializationData {
  llvm::DenseMap<uint32_t, mlir::Value>* mlirValues{};
  llvm::DenseMap<uint32_t, mlir::func::FuncOp>* mlirFuncs{};
  llvm::SmallVector<llvm::StringRef>* strings{};
};

//===----------------------------------------------------------------------===//
// Qubit operations
//===----------------------------------------------------------------------===//

void deserializeQubitAlloc(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                           DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  auto allocOp =
      builder.create<mlir::jeff::QubitAllocOp>(builder.getUnknownLoc());
  mlirValues[operation.getOutputs()[0]] = allocOp.getResult();
}

void deserializeQubitFree(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                          DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  builder.create<mlir::jeff::QubitFreeOp>(builder.getUnknownLoc(),
                                          mlirValues[operation.getInputs()[0]]);
}

void deserializeQubitFreeZero(mlir::OpBuilder& builder,
                              jeff::Op::Reader operation,
                              DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  builder.create<mlir::jeff::QubitFreeZeroOp>(
      builder.getUnknownLoc(), mlirValues[operation.getInputs()[0]]);
}

void deserializeMeasure(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                        DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  auto op = builder.create<mlir::jeff::QubitMeasureOp>(
      builder.getUnknownLoc(), mlirValues[operation.getInputs()[0]]);
  mlirValues[operation.getOutputs()[0]] = op.getResult();
}

void deserializeMeasureNd(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                          DeserializationData& data) {
  const auto outputs = operation.getOutputs();
  auto& mlirValues = *data.mlirValues;
  auto op = builder.create<mlir::jeff::QubitMeasureNDOp>(
      builder.getUnknownLoc(), mlirValues[operation.getInputs()[0]]);
  mlirValues[outputs[0]] = op.getOutQubit();
  mlirValues[outputs[1]] = op.getResult();
}

void deserializeReset(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                      DeserializationData& data) {
  const auto outputs = operation.getOutputs();
  auto& mlirValues = *data.mlirValues;
  auto op = builder.create<mlir::jeff::QubitResetOp>(
      builder.getUnknownLoc(), mlirValues[operation.getInputs()[0]]);
  mlirValues[outputs[0]] = op.getOutQubit();
}

template <typename OpType>
void deserializeOneTargetZeroParameter(mlir::OpBuilder& builder,
                                       jeff::Op::Reader operation,
                                       DeserializationData& data) {
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto& mlirValues = *data.mlirValues;
  const auto gate = operation.getInstruction().getQubit().getGate();
  const auto numControls = gate.getControlQubits();
  llvm::SmallVector<mlir::Value> controls;
  for (uint8_t i = 1; i < 1 + numControls; ++i) {
    controls.push_back(mlirValues[inputs[i]]);
  }
  auto op =
      builder.create<OpType>(builder.getUnknownLoc(), mlirValues[inputs[0]],
                             controls, static_cast<uint8_t>(controls.size()),
                             gate.getAdjoint(), gate.getPower());
  mlirValues[outputs[0]] = op.getOutQubit();
  for (uint8_t i = 1; i < 1 + numControls; ++i) {
    mlirValues[outputs[i]] = op.getOutCtrlQubits()[i - 1];
  }
}

template <typename OpType>
void deserializeOneTargetOneParameter(mlir::OpBuilder& builder,
                                      jeff::Op::Reader operation,
                                      DeserializationData& data) {
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto& mlirValues = *data.mlirValues;
  const auto gate = operation.getInstruction().getQubit().getGate();
  const auto numControls = gate.getControlQubits();
  llvm::SmallVector<mlir::Value> controls;
  for (uint8_t i = 1; i < 1 + numControls; ++i) {
    controls.push_back(mlirValues[inputs[i]]);
  }
  auto rotation = mlirValues[inputs[1 + numControls]];
  auto op = builder.create<OpType>(builder.getUnknownLoc(),
                                   mlirValues[inputs[0]], rotation, controls,
                                   static_cast<uint8_t>(controls.size()),
                                   gate.getAdjoint(), gate.getPower());
  mlirValues[outputs[0]] = op.getOutQubit();
  for (uint8_t i = 1; i < 1 + numControls; ++i) {
    mlirValues[outputs[i]] = op.getOutCtrlQubits()[i - 1];
  }
}

void deserializeU(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                  DeserializationData& data) {
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto& mlirValues = *data.mlirValues;
  const auto gate = operation.getInstruction().getQubit().getGate();
  const auto numControls = gate.getControlQubits();
  llvm::SmallVector<mlir::Value> controls;
  for (uint8_t i = 1; i < 1 + numControls; ++i) {
    controls.push_back(mlirValues[inputs[i]]);
  }
  auto theta = mlirValues[inputs[1 + numControls]];
  auto phi = mlirValues[inputs[2 + numControls]];
  auto lambda = mlirValues[inputs[3 + numControls]];
  auto op = builder.create<mlir::jeff::UOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], theta, phi, lambda,
      controls, static_cast<uint8_t>(controls.size()), gate.getAdjoint(),
      gate.getPower());
  mlirValues[outputs[0]] = op.getOutQubit();
  for (uint8_t i = 1; i < 1 + numControls; ++i) {
    mlirValues[outputs[i]] = op.getOutCtrlQubits()[i - 1];
  }
}

void deserializeSwap(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                     DeserializationData& data) {
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto& mlirValues = *data.mlirValues;
  const auto gate = operation.getInstruction().getQubit().getGate();
  const auto numControls = gate.getControlQubits();
  llvm::SmallVector<mlir::Value> controls;
  for (uint8_t i = 2; i < 2 + numControls; ++i) {
    controls.push_back(mlirValues[inputs[i]]);
  }
  auto op = builder.create<mlir::jeff::SwapOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], mlirValues[inputs[1]],
      controls, static_cast<uint8_t>(controls.size()), gate.getAdjoint(),
      gate.getPower());
  mlirValues[outputs[0]] = op.getOutQubitOne();
  mlirValues[outputs[1]] = op.getOutQubitTwo();
  for (uint8_t i = 2; i < 2 + numControls; ++i) {
    mlirValues[outputs[i]] = op.getOutCtrlQubits()[i - 2];
  }
}

void deserializeGPhase(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                       DeserializationData& data) {
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto& mlirValues = *data.mlirValues;
  const auto gate = operation.getInstruction().getQubit().getGate();
  const auto numControls = gate.getControlQubits();
  llvm::SmallVector<mlir::Value> controls;
  for (uint8_t i = 0; i < numControls; ++i) {
    controls.push_back(mlirValues[inputs[i]]);
  }
  auto rotation = mlirValues[inputs[numControls]];
  auto op = builder.create<mlir::jeff::GPhaseOp>(
      builder.getUnknownLoc(), rotation, controls,
      static_cast<uint8_t>(controls.size()), gate.getAdjoint(),
      gate.getPower());
  for (uint8_t i = 0; i < numControls; ++i) {
    mlirValues[outputs[i]] = op.getOutCtrlQubits()[i];
  }
}

void deserializeWellKnown(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                          DeserializationData& data) {
  const auto wellKnown =
      operation.getInstruction().getQubit().getGate().getWellKnown();
  switch (wellKnown) {
  case jeff::WellKnownGate::X:
    deserializeOneTargetZeroParameter<mlir::jeff::XOp>(builder, operation,
                                                       data);
    break;
  case jeff::WellKnownGate::Y:
    deserializeOneTargetZeroParameter<mlir::jeff::YOp>(builder, operation,
                                                       data);
    break;
  case jeff::WellKnownGate::Z:
    deserializeOneTargetZeroParameter<mlir::jeff::ZOp>(builder, operation,
                                                       data);
    break;
  case jeff::WellKnownGate::S:
    deserializeOneTargetZeroParameter<mlir::jeff::SOp>(builder, operation,
                                                       data);
    break;
  case jeff::WellKnownGate::T:
    deserializeOneTargetZeroParameter<mlir::jeff::TOp>(builder, operation,
                                                       data);
    break;
  case jeff::WellKnownGate::R1:
    deserializeOneTargetOneParameter<mlir::jeff::R1Op>(builder, operation,
                                                       data);
    break;
  case jeff::WellKnownGate::RX:
    deserializeOneTargetOneParameter<mlir::jeff::RxOp>(builder, operation,
                                                       data);
    break;
  case jeff::WellKnownGate::RY:
    deserializeOneTargetOneParameter<mlir::jeff::RyOp>(builder, operation,
                                                       data);
    break;
  case jeff::WellKnownGate::RZ:
    deserializeOneTargetOneParameter<mlir::jeff::RzOp>(builder, operation,
                                                       data);
    break;
  case jeff::WellKnownGate::H:
    deserializeOneTargetZeroParameter<mlir::jeff::HOp>(builder, operation,
                                                       data);
    break;
  case jeff::WellKnownGate::U:
    deserializeU(builder, operation, data);
    break;
  case jeff::WellKnownGate::SWAP:
    deserializeSwap(builder, operation, data);
    break;
  case jeff::WellKnownGate::I:
    deserializeOneTargetZeroParameter<mlir::jeff::IOp>(builder, operation,
                                                       data);
    break;
  case jeff::WellKnownGate::GPHASE:
    deserializeGPhase(builder, operation, data);
    break;
  default:
    llvm::errs() << "Cannot deserialize well-known gate "
                 << static_cast<int>(wellKnown) << "\n";
    llvm::report_fatal_error("Unknown well-known gate");
  }
}

void deserializeCustom(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                       DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  auto& strings = *data.strings;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  const auto gate = operation.getInstruction().getQubit().getGate();
  const auto custom = gate.getCustom();
  const auto name = strings[custom.getName()];
  const auto numControls = gate.getControlQubits();
  const auto numTargets = custom.getNumQubits();
  const auto numQubits = static_cast<uint32_t>(numTargets + numControls);
  const auto numParams = custom.getNumParams();
  llvm::SmallVector<mlir::Value> targets;
  for (uint32_t i = 0; i < numTargets; ++i) {
    targets.push_back(mlirValues[inputs[i]]);
  }
  llvm::SmallVector<mlir::Value> controls;
  for (uint32_t i = numTargets; i < numQubits; ++i) {
    controls.push_back(mlirValues[inputs[i]]);
  }
  llvm::SmallVector<mlir::Value> params;
  for (uint32_t i = numQubits; i < numQubits + numParams; ++i) {
    params.push_back(mlirValues[inputs[i]]);
  }
  auto op = builder.create<mlir::jeff::CustomOp>(
      builder.getUnknownLoc(), targets, controls, params, numControls,
      gate.getAdjoint(), gate.getPower(), name, numTargets, numParams);
  for (uint32_t i = 0; i < numTargets; ++i) {
    mlirValues[outputs[i]] = op.getOutTargetQubits()[i];
  }
  for (uint32_t i = numTargets; i < numQubits; ++i) {
    mlirValues[outputs[i]] = op.getOutCtrlQubits()[i - numTargets];
  }
}

void deserializePpr(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                    DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  const auto gate = operation.getInstruction().getQubit().getGate();
  const auto pauliString = gate.getPpr().getPauliString();
  const auto numTargets = pauliString.size();
  const auto numControls = gate.getControlQubits();
  const auto numQubits = static_cast<uint32_t>(numTargets + numControls);
  llvm::SmallVector<mlir::Value> targets;
  for (uint32_t i = 0; i < numTargets; ++i) {
    targets.push_back(mlirValues[inputs[i]]);
  }
  llvm::SmallVector<mlir::Value> controls;
  for (uint32_t i = numTargets; i < numQubits; ++i) {
    controls.push_back(mlirValues[inputs[i]]);
  }
  auto rotation = mlirValues[inputs[numQubits]];
  llvm::SmallVector<int32_t> pauliStringVector;
  for (auto pauli : pauliString) {
    pauliStringVector.push_back(static_cast<int32_t>(pauli));
  }
  auto pauliStringArrayAttr =
      mlir::DenseI32ArrayAttr::get(builder.getContext(), pauliStringVector);
  auto op = builder.create<mlir::jeff::PPROp>(
      builder.getUnknownLoc(), targets, controls, rotation, numControls,
      gate.getAdjoint(), gate.getPower(), pauliStringArrayAttr);
  for (uint32_t i = 0; i < numTargets; ++i) {
    mlirValues[outputs[i]] = op.getOutQubits()[i];
  }
  for (uint32_t i = numTargets; i < numQubits; ++i) {
    mlirValues[outputs[i]] = op.getOutCtrlQubits()[i - numTargets];
  }
}

void deserializeGate(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                     DeserializationData& data) {
  const auto gate = operation.getInstruction().getQubit().getGate();
  switch (gate.which()) {
  case jeff::QubitGate::WELL_KNOWN:
    deserializeWellKnown(builder, operation, data);
    break;
  case jeff::QubitGate::CUSTOM:
    deserializeCustom(builder, operation, data);
    break;
  case jeff::QubitGate::PPR:
    deserializePpr(builder, operation, data);
    break;
  default:
    llvm::errs() << "Cannot deserialize gate instruction "
                 << static_cast<int>(gate.which()) << "\n";
    llvm::report_fatal_error("Unknown gate instruction");
  }
}

void deserializeQubit(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                      DeserializationData& data) {
  const auto qubit = operation.getInstruction().getQubit();
  switch (qubit.which()) {
  case jeff::QubitOp::ALLOC:
    deserializeQubitAlloc(builder, operation, data);
    break;
  case jeff::QubitOp::FREE:
    deserializeQubitFree(builder, operation, data);
    break;
  case jeff::QubitOp::FREE_ZERO:
    deserializeQubitFreeZero(builder, operation, data);
    break;
  case jeff::QubitOp::MEASURE:
    deserializeMeasure(builder, operation, data);
    break;
  case jeff::QubitOp::MEASURE_ND:
    deserializeMeasureNd(builder, operation, data);
    break;
  case jeff::QubitOp::RESET:
    deserializeReset(builder, operation, data);
    break;
  case jeff::QubitOp::GATE:
    deserializeGate(builder, operation, data);
    break;
  default:
    llvm::errs() << "Cannot deserialize qubit instruction "
                 << static_cast<int>(qubit.which()) << "\n";
    llvm::report_fatal_error("Unknown qubit instruction");
  }
}

//===----------------------------------------------------------------------===//
// Qureg operations
//===----------------------------------------------------------------------===//

void deserializeQuregAlloc(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                           DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  auto allocOp = builder.create<mlir::jeff::QuregAllocOp>(
      builder.getUnknownLoc(), mlirValues[operation.getInputs()[0]]);
  mlirValues[operation.getOutputs()[0]] = allocOp.getResult();
}

void deserializeQuregFreeZero(mlir::OpBuilder& builder,
                              jeff::Op::Reader operation,
                              DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  builder.create<mlir::jeff::QuregFreeZeroOp>(
      builder.getUnknownLoc(), mlirValues[operation.getInputs()[0]]);
}

void deserializeQuregExtractIndex(mlir::OpBuilder& builder,
                                  jeff::Op::Reader operation,
                                  DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::QuregExtractIndexOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], mlirValues[inputs[1]]);
  mlirValues[outputs[0]] = op.getOutQreg();
  mlirValues[outputs[1]] = op.getOutQubit();
}

void deserializeQuregInsertIndex(mlir::OpBuilder& builder,
                                 jeff::Op::Reader operation,
                                 DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::QuregInsertIndexOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], mlirValues[inputs[2]],
      mlirValues[inputs[1]]);
  mlirValues[outputs[0]] = op.getOutQreg();
}

void deserializeQuregExtractSlice(mlir::OpBuilder& builder,
                                  jeff::Op::Reader operation,
                                  DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::QuregExtractSliceOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], mlirValues[inputs[1]],
      mlirValues[inputs[2]]);
  mlirValues[outputs[0]] = op.getOutQreg();
  mlirValues[outputs[1]] = op.getNewQreg();
}

void deserializeQuregInsertSlice(mlir::OpBuilder& builder,
                                 jeff::Op::Reader operation,
                                 DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::QuregInsertSliceOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], mlirValues[inputs[2]],
      mlirValues[inputs[1]]);
  mlirValues[outputs[0]] = op.getOutQreg();
}

void deserializeQuregLength(mlir::OpBuilder& builder,
                            jeff::Op::Reader operation,
                            DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::QuregLengthOp>(builder.getUnknownLoc(),
                                                      mlirValues[inputs[0]]);
  mlirValues[outputs[0]] = op.getOutQreg();
  mlirValues[outputs[1]] = op.getLength();
}

void deserializeQuregSplit(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                           DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::QuregSplitOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], mlirValues[inputs[1]]);
  mlirValues[outputs[0]] = op.getOutQregOne();
  mlirValues[outputs[1]] = op.getOutQregTwo();
}

void deserializeQuregJoin(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                          DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::QuregJoinOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], mlirValues[inputs[1]]);
  mlirValues[outputs[0]] = op.getOutQreg();
}

void deserializeQuregCreate(mlir::OpBuilder& builder,
                            jeff::Op::Reader operation,
                            DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  llvm::SmallVector<mlir::Value> inQreg;
  inQreg.reserve(inputs.size());
  for (auto input : inputs) {
    inQreg.push_back(mlirValues[input]);
  }
  auto op = builder.create<mlir::jeff::QuregCreateOp>(builder.getUnknownLoc(),
                                                      inQreg);
  mlirValues[outputs[0]] = op.getOutQreg();
}

void deserializeQuregFree(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                          DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  builder.create<mlir::jeff::QuregFreeOp>(builder.getUnknownLoc(),
                                          mlirValues[operation.getInputs()[0]]);
}

void deserializeQureg(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                      DeserializationData& data) {
  const auto instruction = operation.getInstruction();
  const auto qureg = instruction.getQureg();
  switch (qureg.which()) {
  case jeff::QuregOp::ALLOC:
    deserializeQuregAlloc(builder, operation, data);
    break;
  case jeff::QuregOp::FREE_ZERO:
    deserializeQuregFreeZero(builder, operation, data);
    break;
  case jeff::QuregOp::EXTRACT_INDEX:
    deserializeQuregExtractIndex(builder, operation, data);
    break;
  case jeff::QuregOp::INSERT_INDEX:
    deserializeQuregInsertIndex(builder, operation, data);
    break;
  case jeff::QuregOp::EXTRACT_SLICE:
    deserializeQuregExtractSlice(builder, operation, data);
    break;
  case jeff::QuregOp::INSERT_SLICE:
    deserializeQuregInsertSlice(builder, operation, data);
    break;
  case jeff::QuregOp::LENGTH:
    deserializeQuregLength(builder, operation, data);
    break;
  case jeff::QuregOp::SPLIT:
    deserializeQuregSplit(builder, operation, data);
    break;
  case jeff::QuregOp::JOIN:
    deserializeQuregJoin(builder, operation, data);
    break;
  case jeff::QuregOp::CREATE:
    deserializeQuregCreate(builder, operation, data);
    break;
  case jeff::QuregOp::FREE:
    deserializeQuregFree(builder, operation, data);
    break;
  default:
    llvm::errs() << "Cannot deserialize qureg instruction "
                 << static_cast<int>(qureg.which()) << "\n";
    llvm::report_fatal_error("Unknown qureg instruction");
  }
}

//===----------------------------------------------------------------------===//
// Int operations
//===----------------------------------------------------------------------===//

#define DESERIALIZE_INT_CONST(BIT_WIDTH)                                       \
  void deserializeIntConst##BIT_WIDTH(mlir::OpBuilder& builder,                \
                                      jeff::Op::Reader operation,              \
                                      DeserializationData& data) {             \
    auto& mlirValues = *data.mlirValues;                                       \
    const auto value =                                                         \
        operation.getInstruction().getInt().getConst##BIT_WIDTH();             \
    auto intType = builder.getI##BIT_WIDTH##Type();                            \
    auto intAttr = mlir::IntegerAttr::get(intType, value);                     \
    auto op = builder.create<mlir::jeff::IntConst##BIT_WIDTH##Op>(             \
        builder.getUnknownLoc(), intType, intAttr);                            \
    mlirValues[operation.getOutputs()[0]] = op.getConstant();                  \
  }

DESERIALIZE_INT_CONST(1)
DESERIALIZE_INT_CONST(8)
DESERIALIZE_INT_CONST(16)
DESERIALIZE_INT_CONST(32)
DESERIALIZE_INT_CONST(64)

#undef DESERIALIZE_INT_CONST

void deserializeIntUnaryOp(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                           mlir::jeff::IntUnaryOperation unaryOperation,
                           DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::IntUnaryOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], unaryOperation);
  mlirValues[outputs[0]] = op.getB();
}

void deserializeIntBinaryOp(mlir::OpBuilder& builder,
                            jeff::Op::Reader operation,
                            mlir::jeff::IntBinaryOperation binaryOperation,
                            DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::IntBinaryOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], mlirValues[inputs[1]],
      binaryOperation);
  mlirValues[outputs[0]] = op.getC();
}

void deserializeIntComparisonOp(
    mlir::OpBuilder& builder, jeff::Op::Reader operation,
    mlir::jeff::IntComparisonOperation comparisonOperation,
    DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::IntComparisonOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], mlirValues[inputs[1]],
      comparisonOperation);
  mlirValues[outputs[0]] = op.getC();
}

#define ADD_CONST_CASE(BIT_WIDTH)                                              \
  case jeff::IntOp::CONST##BIT_WIDTH:                                          \
    deserializeIntConst##BIT_WIDTH(builder, operation, data);                  \
    break;

#define ADD_UNARY_CASE(JEFF_ENUM_VALUE, MLIR_ENUM_SUFFIX)                      \
  case jeff::IntOp::JEFF_ENUM_VALUE:                                           \
    deserializeIntUnaryOp(builder, operation,                                  \
                          mlir::jeff::IntUnaryOperation::_##MLIR_ENUM_SUFFIX,  \
                          data);                                               \
    break;

#define ADD_BINARY_CASE(JEFF_ENUM_VALUE, MLIR_ENUM_SUFFIX)                     \
  case jeff::IntOp::JEFF_ENUM_VALUE:                                           \
    deserializeIntBinaryOp(                                                    \
        builder, operation,                                                    \
        mlir::jeff::IntBinaryOperation::_##MLIR_ENUM_SUFFIX, data);            \
    break;

#define ADD_COMPARISON_CASE(JEFF_ENUM_VALUE, MLIR_ENUM_SUFFIX)                 \
  case jeff::IntOp::JEFF_ENUM_VALUE:                                           \
    deserializeIntComparisonOp(                                                \
        builder, operation,                                                    \
        mlir::jeff::IntComparisonOperation::_##MLIR_ENUM_SUFFIX, data);        \
    break;

void deserializeInt(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                    DeserializationData& data) {
  const auto intInstr = operation.getInstruction().getInt();
  switch (intInstr.which()) {
    ADD_CONST_CASE(1)
    ADD_CONST_CASE(8)
    ADD_CONST_CASE(16)
    ADD_CONST_CASE(32)
    ADD_CONST_CASE(64)
    ADD_UNARY_CASE(NOT, not)
    ADD_UNARY_CASE(ABS, abs)
    ADD_BINARY_CASE(ADD, add)
    ADD_BINARY_CASE(SUB, sub)
    ADD_BINARY_CASE(MUL, mul)
    ADD_BINARY_CASE(DIV_S, divS)
    ADD_BINARY_CASE(DIV_U, divU)
    ADD_BINARY_CASE(POW, pow)
    ADD_BINARY_CASE(AND, and)
    ADD_BINARY_CASE(OR, or)
    ADD_BINARY_CASE(XOR, xor)
    ADD_BINARY_CASE(MIN_S, minS)
    ADD_BINARY_CASE(MIN_U, minU)
    ADD_BINARY_CASE(MAX_S, maxS)
    ADD_BINARY_CASE(MAX_U, maxU)
    ADD_BINARY_CASE(REM_S, remS)
    ADD_BINARY_CASE(REM_U, remU)
    ADD_BINARY_CASE(SHL, shl)
    ADD_BINARY_CASE(SHR, shr)
    ADD_COMPARISON_CASE(EQ, eq)
    ADD_COMPARISON_CASE(LT_S, ltS)
    ADD_COMPARISON_CASE(LTE_S, lteS)
    ADD_COMPARISON_CASE(LT_U, ltU)
    ADD_COMPARISON_CASE(LTE_U, lteU)
  default:
    llvm::errs() << "Cannot deserialize int instruction "
                 << static_cast<int>(intInstr.which()) << "\n";
    llvm::report_fatal_error("Unknown int instruction");
  }
}

#undef ADD_CONST_CASE
#undef ADD_UNARY_CASE
#undef ADD_BINARY_CASE
#undef ADD_COMPARISON_CASE

//===----------------------------------------------------------------------===//
// IntArray operations
//===----------------------------------------------------------------------===//

void deserializeIntArrayConst1(mlir::OpBuilder& builder,
                               jeff::Op::Reader operation,
                               DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto values = operation.getInstruction().getIntArray().getConst1();
  llvm::SmallVector<bool> inArray;
  inArray.reserve(values.size());
  for (auto value : values) {
    inArray.push_back(static_cast<bool>(value));
  }
  auto inArrayAttr =
      mlir::DenseBoolArrayAttr::get(builder.getContext(), inArray);
  auto tensorType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(inArray.size())}, builder.getI1Type());
  auto op = builder.create<mlir::jeff::IntArrayConst1Op>(
      builder.getUnknownLoc(), tensorType, inArrayAttr);
  mlirValues[operation.getOutputs()[0]] = op.getOutArray();
}

#define DESERIALIZE_INT_ARRAY_CONST(BIT_WIDTH)                                 \
  void deserializeIntArrayConst##BIT_WIDTH(mlir::OpBuilder& builder,           \
                                           jeff::Op::Reader operation,         \
                                           DeserializationData& data) {        \
    auto& mlirValues = *data.mlirValues;                                       \
    const auto values =                                                        \
        operation.getInstruction().getIntArray().getConst##BIT_WIDTH();        \
    llvm::SmallVector<int##BIT_WIDTH##_t> inArray;                             \
    inArray.reserve(values.size());                                            \
    for (auto value : values) {                                                \
      inArray.push_back(static_cast<int##BIT_WIDTH##_t>(value));               \
    }                                                                          \
    auto inArrayAttr = mlir::DenseI##BIT_WIDTH##ArrayAttr::get(                \
        builder.getContext(), inArray);                                        \
    auto tensorType =                                                          \
        mlir::RankedTensorType::get({static_cast<int64_t>(inArray.size())},    \
                                    builder.getI##BIT_WIDTH##Type());          \
    auto op = builder.create<mlir::jeff::IntArrayConst##BIT_WIDTH##Op>(        \
        builder.getUnknownLoc(), tensorType, inArrayAttr);                     \
    mlirValues[operation.getOutputs()[0]] = op.getOutArray();                  \
  }

DESERIALIZE_INT_ARRAY_CONST(8)
DESERIALIZE_INT_ARRAY_CONST(16)
DESERIALIZE_INT_ARRAY_CONST(32)
DESERIALIZE_INT_ARRAY_CONST(64)

#undef DESERIALIZE_INT_ARRAY_CONST

void deserializeIntArrayZero(mlir::OpBuilder& builder,
                             jeff::Op::Reader operation,
                             DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  const auto zero = operation.getInstruction().getIntArray().getZero();
  auto tensorType = mlir::RankedTensorType::get({mlir::ShapedType::kDynamic},
                                                builder.getIntegerType(zero));
  auto op = builder.create<mlir::jeff::IntArrayZeroOp>(
      builder.getUnknownLoc(), tensorType, mlirValues[inputs[0]]);
  mlirValues[outputs[0]] = op.getOutArray();
}

void deserializeIntArrayGetIndex(mlir::OpBuilder& builder,
                                 jeff::Op::Reader operation,
                                 DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto tensorType = mlirValues[inputs[0]].getType();
  auto entryType =
      llvm::cast<mlir::RankedTensorType>(tensorType).getElementType();
  auto op = builder.create<mlir::jeff::IntArrayGetIndexOp>(
      builder.getUnknownLoc(), entryType, mlirValues[inputs[0]],
      mlirValues[inputs[1]]);
  mlirValues[outputs[0]] = op.getValue();
}

void deserializeIntArraySetIndex(mlir::OpBuilder& builder,
                                 jeff::Op::Reader operation,
                                 DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto tensorType = mlirValues[inputs[0]].getType();
  auto op = builder.create<mlir::jeff::IntArraySetIndexOp>(
      builder.getUnknownLoc(), tensorType, mlirValues[inputs[0]],
      mlirValues[inputs[1]], mlirValues[inputs[2]]);
  mlirValues[outputs[0]] = op.getOutArray();
}

void deserializeIntArrayLength(mlir::OpBuilder& builder,
                               jeff::Op::Reader operation,
                               DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::IntArrayLengthOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]]);
  mlirValues[outputs[0]] = op.getLength();
}

void deserializeIntArrayCreate(mlir::OpBuilder& builder,
                               jeff::Op::Reader operation,
                               DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  llvm::SmallVector<mlir::Value> inArray;
  inArray.reserve(inputs.size());
  for (auto input : inputs) {
    inArray.push_back(mlirValues[input]);
  }
  auto tensorType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(inputs.size())}, mlirValues[inputs[0]].getType());
  auto op = builder.create<mlir::jeff::IntArrayCreateOp>(
      builder.getUnknownLoc(), tensorType, inArray);
  mlirValues[outputs[0]] = op.getOutArray();
}

void deserializeIntArray(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                         DeserializationData& data) {
  const auto intArray = operation.getInstruction().getIntArray();
  switch (intArray.which()) {
  case jeff::IntArrayOp::CONST1:
    deserializeIntArrayConst1(builder, operation, data);
    break;
  case jeff::IntArrayOp::CONST8:
    deserializeIntArrayConst8(builder, operation, data);
    break;
  case jeff::IntArrayOp::CONST16:
    deserializeIntArrayConst16(builder, operation, data);
    break;
  case jeff::IntArrayOp::CONST32:
    deserializeIntArrayConst32(builder, operation, data);
    break;
  case jeff::IntArrayOp::CONST64:
    deserializeIntArrayConst64(builder, operation, data);
    break;
  case jeff::IntArrayOp::ZERO:
    deserializeIntArrayZero(builder, operation, data);
    break;
  case jeff::IntArrayOp::GET_INDEX:
    deserializeIntArrayGetIndex(builder, operation, data);
    break;
  case jeff::IntArrayOp::SET_INDEX:
    deserializeIntArraySetIndex(builder, operation, data);
    break;
  case jeff::IntArrayOp::LENGTH:
    deserializeIntArrayLength(builder, operation, data);
    break;
  case jeff::IntArrayOp::CREATE:
    deserializeIntArrayCreate(builder, operation, data);
    break;
  default:
    llvm::errs() << "Cannot deserialize int array instruction "
                 << static_cast<int>(intArray.which()) << "\n";
    llvm::report_fatal_error("Unknown int array instruction");
  }
}

//===----------------------------------------------------------------------===//
// Float operations
//===----------------------------------------------------------------------===//

#define DESERIALIZE_FLOAT_CONST(BIT_WIDTH)                                     \
  void deserializeFloatConst##BIT_WIDTH(mlir::OpBuilder& builder,              \
                                        jeff::Op::Reader operation,            \
                                        DeserializationData& data) {           \
    auto& mlirValues = *data.mlirValues;                                       \
    const auto value =                                                         \
        operation.getInstruction().getFloat().getConst##BIT_WIDTH();           \
    auto floatType = builder.getF##BIT_WIDTH##Type();                          \
    auto floatAttr = mlir::FloatAttr::get(floatType, value);                   \
    auto op = builder.create<mlir::jeff::FloatConst##BIT_WIDTH##Op>(           \
        builder.getUnknownLoc(), floatType, floatAttr);                        \
    mlirValues[operation.getOutputs()[0]] = op.getConstant();                  \
  }

DESERIALIZE_FLOAT_CONST(32)
DESERIALIZE_FLOAT_CONST(64)

#undef DESERIALIZE_FLOAT_CONST

void deserializeFloatUnaryOp(mlir::OpBuilder& builder,
                             jeff::Op::Reader operation,
                             mlir::jeff::FloatUnaryOperation unaryOperation,
                             DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::FloatUnaryOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], unaryOperation);
  mlirValues[outputs[0]] = op.getB();
}

void deserializeFloatBinaryOp(mlir::OpBuilder& builder,
                              jeff::Op::Reader operation,
                              mlir::jeff::FloatBinaryOperation binaryOperation,
                              DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::FloatBinaryOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], mlirValues[inputs[1]],
      binaryOperation);
  mlirValues[outputs[0]] = op.getC();
}

void deserializeFloatComparisonOp(
    mlir::OpBuilder& builder, jeff::Op::Reader operation,
    mlir::jeff::FloatComparisonOperation comparisonOperation,
    DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::FloatComparisonOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], mlirValues[inputs[1]],
      comparisonOperation);
  mlirValues[outputs[0]] = op.getC();
}

void deserializeFloatIsOp(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                          mlir::jeff::FloatIsOperation isOperation,
                          DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::FloatIsOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], isOperation);
  mlirValues[outputs[0]] = op.getResult();
}

#define ADD_CONST_CASE(BIT_WIDTH)                                              \
  case jeff::FloatOp::CONST##BIT_WIDTH:                                        \
    deserializeFloatConst##BIT_WIDTH(builder, operation, data);                \
    break;

#define ADD_UNARY_CASE(JEFF_ENUM_VALUE, MLIR_ENUM_SUFFIX)                      \
  case jeff::FloatOp::JEFF_ENUM_VALUE:                                         \
    deserializeFloatUnaryOp(                                                   \
        builder, operation,                                                    \
        mlir::jeff::FloatUnaryOperation::_##MLIR_ENUM_SUFFIX, data);           \
    break;

#define ADD_BINARY_CASE(JEFF_ENUM_VALUE, MLIR_ENUM_SUFFIX)                     \
  case jeff::FloatOp::JEFF_ENUM_VALUE:                                         \
    deserializeFloatBinaryOp(                                                  \
        builder, operation,                                                    \
        mlir::jeff::FloatBinaryOperation::_##MLIR_ENUM_SUFFIX, data);          \
    break;

#define ADD_COMPARISON_CASE(JEFF_ENUM_VALUE, MLIR_ENUM_SUFFIX)                 \
  case jeff::FloatOp::JEFF_ENUM_VALUE:                                         \
    deserializeFloatComparisonOp(                                              \
        builder, operation,                                                    \
        mlir::jeff::FloatComparisonOperation::_##MLIR_ENUM_SUFFIX, data);      \
    break;

#define ADD_IS_CASE(JEFF_ENUM_VALUE, MLIR_ENUM_SUFFIX)                         \
  case jeff::FloatOp::JEFF_ENUM_VALUE:                                         \
    deserializeFloatIsOp(builder, operation,                                   \
                         mlir::jeff::FloatIsOperation::_is##MLIR_ENUM_SUFFIX,  \
                         data);                                                \
    break;

void deserializeFloat(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                      DeserializationData& data) {
  const auto floatInstr = operation.getInstruction().getFloat();
  switch (floatInstr.which()) {
    ADD_CONST_CASE(32)
    ADD_CONST_CASE(64)
    ADD_UNARY_CASE(SQRT, sqrt)
    ADD_UNARY_CASE(ABS, abs)
    ADD_UNARY_CASE(CEIL, ceil)
    ADD_UNARY_CASE(FLOOR, floor)
    ADD_UNARY_CASE(EXP, exp)
    ADD_UNARY_CASE(LOG, log)
    ADD_UNARY_CASE(SIN, sin)
    ADD_UNARY_CASE(COS, cos)
    ADD_UNARY_CASE(TAN, tan)
    ADD_UNARY_CASE(ASIN, asin)
    ADD_UNARY_CASE(ACOS, acos)
    ADD_UNARY_CASE(ATAN, atan)
    ADD_UNARY_CASE(SINH, sinh)
    ADD_UNARY_CASE(COSH, cosh)
    ADD_UNARY_CASE(TANH, tanh)
    ADD_UNARY_CASE(ASINH, asinh)
    ADD_UNARY_CASE(ACOSH, acosh)
    ADD_UNARY_CASE(ATANH, atanh)
    ADD_BINARY_CASE(ADD, add)
    ADD_BINARY_CASE(SUB, sub)
    ADD_BINARY_CASE(MUL, mul)
    ADD_BINARY_CASE(POW, pow)
    ADD_BINARY_CASE(ATAN2, atan2)
    ADD_BINARY_CASE(MAX, max)
    ADD_BINARY_CASE(MIN, min)
    ADD_COMPARISON_CASE(EQ, eq)
    ADD_COMPARISON_CASE(LT, lt)
    ADD_COMPARISON_CASE(LTE, lte)
    ADD_IS_CASE(IS_NAN, Nan)
    ADD_IS_CASE(IS_INF, Inf)
  default:
    llvm::errs() << "Cannot deserialize float instruction "
                 << static_cast<int>(floatInstr.which()) << "\n";
    llvm::report_fatal_error("Unknown float instruction");
  }
}

#undef ADD_CONST_CASE
#undef ADD_UNARY_CASE
#undef ADD_BINARY_CASE
#undef ADD_COMPARISON_CASE
#undef ADD_IS_CASE

//===----------------------------------------------------------------------===//
// FloatArray operations
//===----------------------------------------------------------------------===//

void deserializeFloatArrayConst32(mlir::OpBuilder& builder,
                                  jeff::Op::Reader operation,
                                  DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto values = operation.getInstruction().getFloatArray().getConst32();
  llvm::SmallVector<float> inArray;
  inArray.reserve(values.size());
  for (auto value : values) {
    inArray.push_back(static_cast<float>(value));
  }
  auto inArrayAttr =
      mlir::DenseF32ArrayAttr::get(builder.getContext(), inArray);
  auto tensorType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(inArray.size())}, builder.getF32Type());
  auto op = builder.create<mlir::jeff::FloatArrayConst32Op>(
      builder.getUnknownLoc(), tensorType, inArrayAttr);
  mlirValues[operation.getOutputs()[0]] = op.getOutArray();
}

void deserializeFloatArrayConst64(mlir::OpBuilder& builder,
                                  jeff::Op::Reader operation,
                                  DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto values = operation.getInstruction().getFloatArray().getConst64();
  llvm::SmallVector<double> inArray;
  inArray.reserve(values.size());
  for (auto value : values) {
    inArray.push_back(static_cast<double>(value));
  }
  auto inArrayAttr =
      mlir::DenseF64ArrayAttr::get(builder.getContext(), inArray);
  auto tensorType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(inArray.size())}, builder.getF64Type());
  auto op = builder.create<mlir::jeff::FloatArrayConst64Op>(
      builder.getUnknownLoc(), tensorType, inArrayAttr);
  mlirValues[operation.getOutputs()[0]] = op.getOutArray();
}

void deserializeFloatArrayZero(mlir::OpBuilder& builder,
                               jeff::Op::Reader operation,
                               DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  const auto zero = operation.getInstruction().getFloatArray().getZero();
  mlir::Type floatType;
  switch (zero) {
  case jeff::FloatPrecision::FLOAT32:
    floatType = builder.getF32Type();
    break;
  case jeff::FloatPrecision::FLOAT64:
    floatType = builder.getF64Type();
    break;
  default:
    llvm::report_fatal_error("Invalid bit width");
  }
  auto tensorType =
      mlir::RankedTensorType::get({mlir::ShapedType::kDynamic}, floatType);
  auto op = builder.create<mlir::jeff::FloatArrayZeroOp>(
      builder.getUnknownLoc(), tensorType, mlirValues[inputs[0]]);
  mlirValues[outputs[0]] = op.getOutArray();
}

void deserializeFloatArrayGetIndex(mlir::OpBuilder& builder,
                                   jeff::Op::Reader operation,
                                   DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto tensorType = mlirValues[inputs[0]].getType();
  auto entryType =
      llvm::cast<mlir::RankedTensorType>(tensorType).getElementType();
  auto op = builder.create<mlir::jeff::FloatArrayGetIndexOp>(
      builder.getUnknownLoc(), entryType, mlirValues[inputs[0]],
      mlirValues[inputs[1]]);
  mlirValues[outputs[0]] = op.getValue();
}

void deserializeFloatArraySetIndex(mlir::OpBuilder& builder,
                                   jeff::Op::Reader operation,
                                   DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto tensorType = mlirValues[inputs[0]].getType();
  auto op = builder.create<mlir::jeff::FloatArraySetIndexOp>(
      builder.getUnknownLoc(), tensorType, mlirValues[inputs[0]],
      mlirValues[inputs[1]], mlirValues[inputs[2]]);
  mlirValues[outputs[0]] = op.getOutArray();
}

void deserializeFloatArrayLength(mlir::OpBuilder& builder,
                                 jeff::Op::Reader operation,
                                 DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto op = builder.create<mlir::jeff::FloatArrayLengthOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]]);
  mlirValues[outputs[0]] = op.getLength();
}

void deserializeFloatArrayCreate(mlir::OpBuilder& builder,
                                 jeff::Op::Reader operation,
                                 DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  llvm::SmallVector<mlir::Value> inArray;
  inArray.reserve(inputs.size());
  for (auto input : inputs) {
    inArray.push_back(mlirValues[input]);
  }
  auto tensorType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(inputs.size())}, mlirValues[inputs[0]].getType());
  auto op = builder.create<mlir::jeff::FloatArrayCreateOp>(
      builder.getUnknownLoc(), tensorType, inArray);
  mlirValues[outputs[0]] = op.getOutArray();
}

void deserializeFloatArray(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                           DeserializationData& data) {
  const auto floatArray = operation.getInstruction().getFloatArray();
  switch (floatArray.which()) {
  case jeff::FloatArrayOp::CONST32:
    deserializeFloatArrayConst32(builder, operation, data);
    break;
  case jeff::FloatArrayOp::CONST64:
    deserializeFloatArrayConst64(builder, operation, data);
    break;
  case jeff::FloatArrayOp::ZERO:
    deserializeFloatArrayZero(builder, operation, data);
    break;
  case jeff::FloatArrayOp::GET_INDEX:
    deserializeFloatArrayGetIndex(builder, operation, data);
    break;
  case jeff::FloatArrayOp::SET_INDEX:
    deserializeFloatArraySetIndex(builder, operation, data);
    break;
  case jeff::FloatArrayOp::LENGTH:
    deserializeFloatArrayLength(builder, operation, data);
    break;
  case jeff::FloatArrayOp::CREATE:
    deserializeFloatArrayCreate(builder, operation, data);
    break;
  default:
    llvm::errs() << "Cannot deserialize float array instruction "
                 << static_cast<int>(floatArray.which()) << "\n";
    llvm::report_fatal_error("Unknown float array instruction");
  }
}

//===----------------------------------------------------------------------===//
// SCF operations
//===----------------------------------------------------------------------===//

// Forward declaration
void deserializeOperations(
    mlir::OpBuilder& builder,
    capnp::List<jeff::Op, capnp::Kind::STRUCT>::Reader operations,
    DeserializationData& data);

void deserializeSwitch(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                       DeserializationData& data) {
  auto loc = builder.getUnknownLoc();
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto switchInstr = operation.getInstruction().getScf().getSwitch();
  const auto branches = switchInstr.getBranches();

  llvm::SmallVector<mlir::Value> inValues;
  llvm::SmallVector<mlir::Type> outTypes;
  inValues.reserve(inputs.size() - 1);
  outTypes.reserve(inputs.size() - 1);
  for (size_t i = 1; i < inputs.size(); ++i) {
    inValues.push_back(mlirValues[inputs[i]]);
    outTypes.push_back(mlirValues[inputs[i]].getType());
  }

  auto op = builder.create<mlir::jeff::SwitchOp>(
      loc, outTypes, mlirValues[inputs[0]], inValues, branches.size());

  for (size_t i = 0; i < branches.size(); ++i) {
    auto& block = op.getBranches()[i].emplaceBlock();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);

    const auto& branch = branches[i];

    // Add sources to mlirValues
    for (size_t j = 0; j < branch.getSources().size(); ++j) {
      auto arg = block.addArgument(inValues[j].getType(), loc);
      mlirValues[branch.getSources()[j]] = arg;
    }

    deserializeOperations(builder, branches[i].getOperations(), data);

    // Retrieve target values
    llvm::SmallVector<mlir::Value> targetValues;
    targetValues.reserve(branch.getTargets().size());
    for (size_t j = 0; j < branch.getTargets().size(); ++j) {
      targetValues.push_back(mlirValues[branch.getTargets()[j]]);
    }

    builder.create<mlir::jeff::YieldOp>(loc, targetValues);
  }

  if (switchInstr.hasDefault()) {
    auto& block = op.getDefault().emplaceBlock();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);

    const auto& defaultRegion = switchInstr.getDefault();

    // Add sources to mlirValues
    for (size_t i = 0; i < defaultRegion.getSources().size(); ++i) {
      auto arg = block.addArgument(inValues[i].getType(), loc);
      mlirValues[defaultRegion.getSources()[i]] = arg;
    }

    deserializeOperations(builder, defaultRegion.getOperations(), data);

    // Retrieve target values
    llvm::SmallVector<mlir::Value> targetValues;
    targetValues.reserve(defaultRegion.getTargets().size());
    for (size_t i = 0; i < defaultRegion.getTargets().size(); ++i) {
      targetValues.push_back(mlirValues[defaultRegion.getTargets()[i]]);
    }

    builder.create<mlir::jeff::YieldOp>(loc, targetValues);
  }

  llvm::SmallVector<mlir::Value> outValues;
  outValues.reserve(operation.getOutputs().size());
  for (size_t i = 0; i < operation.getOutputs().size(); ++i) {
    mlirValues[operation.getOutputs()[i]] = op.getResults()[i];
  }
}

void deserializeFor(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                    DeserializationData& data) {
  auto loc = builder.getUnknownLoc();
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto forInstr = operation.getInstruction().getScf().getFor();

  llvm::SmallVector<mlir::Value> inValues;
  llvm::SmallVector<mlir::Type> outTypes;
  inValues.reserve(inputs.size() - 3);
  outTypes.reserve(inputs.size() - 3);
  for (size_t i = 3; i < inputs.size(); ++i) {
    inValues.push_back(mlirValues[inputs[i]]);
    outTypes.push_back(mlirValues[inputs[i]].getType());
  }

  auto op = builder.create<mlir::jeff::ForOp>(
      loc, outTypes, mlirValues[inputs[0]], mlirValues[inputs[1]],
      mlirValues[inputs[2]], inValues);

  {
    auto& bodyBlock = op.getBody().emplaceBlock();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);

    // Add induction variable to mlirValues
    auto i = bodyBlock.addArgument(mlirValues[inputs[0]].getType(), loc);
    mlirValues[forInstr.getSources()[0]] = i;

    // Add sources to mlirValues
    for (size_t i = 1; i < forInstr.getSources().size(); ++i) {
      auto arg = bodyBlock.addArgument(inValues[i - 1].getType(), loc);
      mlirValues[forInstr.getSources()[i]] = arg;
    }

    deserializeOperations(builder, forInstr.getOperations(), data);

    llvm::SmallVector<mlir::Value> outValues;
    outValues.reserve(forInstr.getTargets().size());
    for (size_t i = 0; i < forInstr.getTargets().size(); ++i) {
      outValues.push_back(mlirValues[forInstr.getTargets()[i]]);
    }

    builder.create<mlir::jeff::YieldOp>(loc, outValues);
  }

  llvm::SmallVector<mlir::Value> outValues;
  outValues.reserve(operation.getOutputs().size());
  for (size_t i = 0; i < operation.getOutputs().size(); ++i) {
    mlirValues[operation.getOutputs()[i]] = op.getResults()[i];
  }
}

template <typename MLIR_WHILE_OP_TYPE, typename JEFF_WHILE_OP_READER_TYPE>
void deserializeWhile(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                      JEFF_WHILE_OP_READER_TYPE reader,
                      DeserializationData& data) {
  auto loc = builder.getUnknownLoc();
  auto& mlirValues = *data.mlirValues;
  const auto inputs = operation.getInputs();

  llvm::SmallVector<mlir::Value> inValues;
  llvm::SmallVector<mlir::Type> outTypes;
  inValues.reserve(inputs.size());
  outTypes.reserve(inputs.size());
  for (const auto input : inputs) {
    inValues.push_back(mlirValues[input]);
    outTypes.push_back(mlirValues[input].getType());
  }

  auto op = builder.create<MLIR_WHILE_OP_TYPE>(loc, outTypes, inValues);

  {
    auto& block = op.getCondition().emplaceBlock();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);

    const auto condition = reader.getCondition();

    // Add sources to mlirValues
    for (size_t i = 0; i < condition.getSources().size(); ++i) {
      auto arg = block.addArgument(inValues[i].getType(), loc);
      mlirValues[condition.getSources()[i]] = arg;
    }

    deserializeOperations(builder, condition.getOperations(), data);

    auto result = mlirValues[condition.getTargets()[0]];
    builder.create<mlir::jeff::YieldOp>(loc, result);
  }

  {
    auto& block = op.getBody().emplaceBlock();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);

    const auto body = reader.getBody();

    // Add sources to mlirValues
    for (size_t i = 0; i < body.getSources().size(); ++i) {
      auto arg = block.addArgument(inValues[i].getType(), loc);
      mlirValues[body.getSources()[i]] = arg;
    }

    deserializeOperations(builder, body.getOperations(), data);

    // Retrieve target values
    llvm::SmallVector<mlir::Value> targetValues;
    targetValues.reserve(body.getTargets().size());
    for (size_t i = 0; i < body.getTargets().size(); ++i) {
      targetValues.push_back(mlirValues[body.getTargets()[i]]);
    }

    builder.create<mlir::jeff::YieldOp>(loc, targetValues);
  }

  llvm::SmallVector<mlir::Value> outValues;
  outValues.reserve(operation.getOutputs().size());
  for (size_t i = 0; i < operation.getOutputs().size(); ++i) {
    mlirValues[operation.getOutputs()[i]] = op.getResults()[i];
  }
}

void deserializeScf(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                    DeserializationData& data) {
  const auto scf = operation.getInstruction().getScf();
  switch (scf.which()) {
  case jeff::ScfOp::SWITCH:
    deserializeSwitch(builder, operation, data);
    break;
  case jeff::ScfOp::FOR:
    deserializeFor(builder, operation, data);
    break;
  case jeff::ScfOp::WHILE:
    deserializeWhile<mlir::jeff::WhileOp, jeff::ScfOp::While::Reader>(
        builder, operation, scf.getWhile(), data);
    break;
  case jeff::ScfOp::DO_WHILE:
    deserializeWhile<mlir::jeff::DoWhileOp, jeff::ScfOp::DoWhile::Reader>(
        builder, operation, scf.getDoWhile(), data);
    break;
  default:
    llvm::errs() << "Cannot deserialize scf instruction "
                 << static_cast<int>(scf.which()) << "\n";
    llvm::report_fatal_error("Unknown scf instruction");
  }
}

//===----------------------------------------------------------------------===//
// Func operations
//===----------------------------------------------------------------------===//

void deserializeFunc(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                     DeserializationData& data) {
  auto& mlirValues = *data.mlirValues;
  auto& mlirFuncs = *data.mlirFuncs;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  const auto func = operation.getInstruction().getFunc();
  auto mlirFunc = mlirFuncs[func.getFuncCall()];
  llvm::SmallVector<mlir::Value> mlirInputs;
  mlirInputs.reserve(inputs.size());
  for (const auto input : inputs) {
    mlirInputs.push_back(mlirValues[input]);
  }
  auto op = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(),
                                               mlirFunc, mlirInputs);
  for (size_t i = 0; i < outputs.size(); ++i) {
    mlirValues[outputs[i]] = op.getResult(i);
  }
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

mlir::Type deserializeIntType(mlir::OpBuilder& builder,
                              jeff::Type::Reader type) {
  switch (type.getInt()) {
  case 1:
    return builder.getI1Type();
  case 8:
    return builder.getI8Type();
  case 16:
    return builder.getI16Type();
  case 32:
    return builder.getI32Type();
  case 64:
    return builder.getI64Type();
  default:
    llvm::errs() << "Cannot deserialize int type "
                 << static_cast<int>(type.getInt()) << "\n";
    llvm::report_fatal_error("Unknown int type");
  }
}

mlir::Type deserializeIntArrayType(mlir::OpBuilder& builder,
                                   jeff::Type::Reader type) {
  switch (type.getIntArray()) {
  case 1:
    return mlir::RankedTensorType::get({mlir::ShapedType::kDynamic},
                                       builder.getI1Type());
  case 8:
    return mlir::RankedTensorType::get({mlir::ShapedType::kDynamic},
                                       builder.getI8Type());
  case 16:
    return mlir::RankedTensorType::get({mlir::ShapedType::kDynamic},
                                       builder.getI16Type());
  case 32:
    return mlir::RankedTensorType::get({mlir::ShapedType::kDynamic},
                                       builder.getI32Type());
  case 64:
    return mlir::RankedTensorType::get({mlir::ShapedType::kDynamic},
                                       builder.getI64Type());
  default:
    llvm::errs() << "Cannot deserialize int array type "
                 << static_cast<int>(type.getIntArray()) << "\n";
    llvm::report_fatal_error("Unknown int array type");
  }
}

mlir::Type deserializeFloatType(mlir::OpBuilder& builder,
                                jeff::Type::Reader type) {
  switch (type.getFloat()) {
  case jeff::FloatPrecision::FLOAT32:
    return builder.getF32Type();
  case jeff::FloatPrecision::FLOAT64:
    return builder.getF64Type();
  default:
    llvm::errs() << "Cannot deserialize float type "
                 << static_cast<int>(type.getFloat()) << "\n";
    llvm::report_fatal_error("Unknown float type");
  }
}

mlir::Type deserializeFloatArrayType(mlir::OpBuilder& builder,
                                     jeff::Type::Reader type) {
  switch (type.getFloatArray()) {
  case jeff::FloatPrecision::FLOAT32:
    return mlir::RankedTensorType::get({mlir::ShapedType::kDynamic},
                                       builder.getF32Type());
  case jeff::FloatPrecision::FLOAT64:
    return mlir::RankedTensorType::get({mlir::ShapedType::kDynamic},
                                       builder.getF64Type());
  default:
    llvm::errs() << "Cannot deserialize float array type "
                 << static_cast<int>(type.getFloatArray()) << "\n";
    llvm::report_fatal_error("Unknown float array type");
  }
}

mlir::Type deserializeType(mlir::OpBuilder& builder, jeff::Type::Reader type) {
  switch (type.which()) {
  case jeff::Type::QUBIT:
    return mlir::jeff::QubitType::get(builder.getContext());
  case jeff::Type::QUREG:
    return mlir::jeff::QuregType::get(builder.getContext());
  case jeff::Type::INT:
    return deserializeIntType(builder, type);
  case jeff::Type::INT_ARRAY:
    return deserializeIntArrayType(builder, type);
  case jeff::Type::FLOAT:
    return deserializeFloatType(builder, type);
  case jeff::Type::FLOAT_ARRAY:
    return deserializeFloatArrayType(builder, type);
  default:
    llvm::errs() << "Cannot deserialize type " << static_cast<int>(type.which())
                 << "\n";
    llvm::report_fatal_error("Unknown type");
  }
}

void deserializeOperations(
    mlir::OpBuilder& builder,
    capnp::List<jeff::Op, capnp::Kind::STRUCT>::Reader operations,
    DeserializationData& data) {
  for (auto operation : operations) {
    const auto instruction = operation.getInstruction();
    switch (instruction.which()) {
    case jeff::Op::Instruction::QUBIT:
      deserializeQubit(builder, operation, data);
      break;
    case jeff::Op::Instruction::QUREG:
      deserializeQureg(builder, operation, data);
      break;
    case jeff::Op::Instruction::INT:
      deserializeInt(builder, operation, data);
      break;
    case jeff::Op::Instruction::INT_ARRAY:
      deserializeIntArray(builder, operation, data);
      break;
    case jeff::Op::Instruction::FLOAT:
      deserializeFloat(builder, operation, data);
      break;
    case jeff::Op::Instruction::FLOAT_ARRAY:
      deserializeFloatArray(builder, operation, data);
      break;
    case jeff::Op::Instruction::SCF:
      deserializeScf(builder, operation, data);
      break;
    case jeff::Op::Instruction::FUNC:
      deserializeFunc(builder, operation, data);
      break;
    default:
      llvm::errs() << "Cannot deserialize instruction "
                   << static_cast<int>(instruction.which()) << "\n";
      llvm::report_fatal_error("Unknown instruction");
    }
  }
}

void deserializeFunction(
    mlir::OpBuilder& builder, jeff::Function::Reader function,
    jeff::Module::Reader jeffModule,
    llvm::DenseMap<uint32_t, mlir::func::FuncOp>& mlirFuncs,
    llvm::SmallVector<llvm::StringRef>& mlirStrings) {

  // Get entry point
  const auto entryPoint = jeffModule.getEntrypoint();

  // Get function definition
  const auto definition = function.getDefinition();

  // Get values
  const auto jeffValues = definition.getValues();

  llvm::DenseMap<uint32_t, mlir::Value> mlirValues;
  mlirValues.reserve(jeffValues.size());

  // Get function body
  if (!definition.hasBody()) {
    llvm::report_fatal_error("Function definition has no body");
  }
  const auto body = definition.getBody();

  // Get operations
  if (!body.hasOperations()) {
    llvm::report_fatal_error("Function body has no operations");
  }
  const auto operations = body.getOperations();

  // Get sources
  const auto sources = body.getSources();

  // Get source types
  llvm::SmallVector<mlir::Type> sourceTypes;
  sourceTypes.reserve(sources.size());
  for (const auto source : sources) {
    const auto jeffType = jeffValues[source].getType();
    sourceTypes.push_back(deserializeType(builder, jeffType));
  }

  // Get targets
  const auto targets = body.getTargets();

  // Get target types
  llvm::SmallVector<mlir::Type> targetTypes;
  targetTypes.reserve(targets.size());
  for (auto target : targets) {
    const auto jeffType = jeffValues[target].getType();
    targetTypes.push_back(deserializeType(builder, jeffType));
  }

  // Create function
  const auto funcName = mlirStrings[function.getName()];
  auto funcType = builder.getFunctionType(sourceTypes, targetTypes);
  auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                 funcName, funcType);
  mlirFuncs[function.getName()] = func;

  // Add attributes if the function is the entry point
  if (function.getName() == entryPoint) {
    llvm::SmallVector<mlir::Attribute> attributes;
    attributes.emplace_back(builder.getStringAttr("entry_point"));
    func->setAttr("passthrough", builder.getArrayAttr(attributes));
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  auto& entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Add sources to mlirValues
  for (auto i = 0; i < sources.size(); ++i) {
    mlirValues[sources[i]] = entryBlock.getArgument(i);
  }

  auto data = DeserializationData{&mlirValues, &mlirFuncs, &mlirStrings};
  deserializeOperations(builder, operations, data);

  llvm::SmallVector<mlir::Value> results;
  results.reserve(body.getTargets().size());
  for (const auto target : body.getTargets()) {
    results.push_back(mlirValues[target]);
  }

  // Create return statement
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), results);
}

} // namespace

mlir::OwningOpRef<mlir::ModuleOp> deserialize(mlir::MLIRContext* context,
                                              const std::string& path) {
  // Get Jeff module from file
  const auto fd = open(path.c_str(), O_RDONLY, 0);
  if (fd < 0) {
    llvm::report_fatal_error("Could not open file");
  }
  capnp::StreamFdMessageReader message(fd);
  jeff::Module::Reader jeffModule = message.getRoot<jeff::Module>();

  // Create MLIR builder
  mlir::OpBuilder builder(context);

  // Create MLIR module
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto mlirModule = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(mlirModule.getBody());

  // Get strings
  auto strings = jeffModule.getStrings();
  auto mlirStrings = llvm::SmallVector<llvm::StringRef>();
  mlirStrings.reserve(strings.size());
  for (auto i = 0; i < strings.size(); ++i) {
    mlirStrings.push_back(strings[i].cStr());
  }

  // Get functions
  if (!jeffModule.hasFunctions()) {
    llvm::report_fatal_error("No functions found in module");
  }
  const auto functions = jeffModule.getFunctions();

  llvm::DenseMap<uint32_t, mlir::func::FuncOp> mlirFuncs;
  mlirFuncs.reserve(functions.size());

  for (const auto function : functions) {
    deserializeFunction(builder, function, jeffModule, mlirFuncs, mlirStrings);
  }

  // Set metadata
  mlirModule->setAttr("jeff.strings", builder.getStrArrayAttr(mlirStrings));
  const auto jeffVersion = std::to_string(jeffModule.getVersion());
  mlirModule->setAttr("jeff.version", builder.getStringAttr(jeffVersion));

  return mlirModule;
}
