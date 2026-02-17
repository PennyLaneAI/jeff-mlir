#include "jeff/IR/JeffDialect.h"
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
  llvm::DenseMap<mlir::Value, uint32_t> valueMap;
  llvm::DenseMap<mlir::func::FuncOp, uint32_t> funcMap;
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
  customBuilder.setName(ctx.strings[op.getName().str()]);
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

void serializeIntConst32(jeff::Op::Builder builder, mlir::jeff::IntConst32Op op,
                         SerializationContext& ctx) {
  auto intBuilder = builder.initInstruction().initInt();
  intBuilder.setConst32(op.getVal());

  builder.initInputs(0);

  auto outputs = builder.initOutputs(1);
  outputs.set(0, ctx.getValueId(op.getConstant()));
}

void serializeInt(jeff::Op::Builder builder, mlir::Operation* operation,
                  SerializationContext& ctx) {
  if (auto op = llvm::dyn_cast<mlir::jeff::IntConst32Op>(operation)) {
    serializeIntConst32(builder, op, ctx);
  } else {
    llvm::errs() << "Cannot serialize int operation " << operation->getName()
                 << "\n";
    llvm::report_fatal_error("Unknown int operation");
  }
}

//===----------------------------------------------------------------------===//
// Float operations
//===----------------------------------------------------------------------===//

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

void serializeFloat(jeff::Op::Builder builder, mlir::Operation* operation,
                    SerializationContext& ctx) {
  if (auto op = llvm::dyn_cast<mlir::jeff::FloatConst64Op>(operation)) {
    serializeFloatConst64(builder, op, ctx);
  } else {
    llvm::errs() << "Cannot serialize float operation " << operation->getName()
                 << "\n";
    llvm::report_fatal_error("Unknown float operation");
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
  } else if (llvm::isa<mlir::jeff::FloatOperation>(operation)) {
    serializeFloat(builder, operation, ctx);
  } else {
    llvm::errs() << "Cannot serialize operation " << operation->getName()
                 << "\n";
    llvm::report_fatal_error("Unknown operation");
  }
}

void collectValues(mlir::func::FuncOp func, SerializationContext& ctx,
                   llvm::SmallVector<mlir::Value>& values) {
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
                       mlir::func::FuncOp func, SerializationContext& ctx) {
  auto defBuilder = funcBuilder.initDefinition();
  auto& entryBlock = func.getRegion().front();

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

  // Set sources
  const auto numSources = entryBlock.getNumArguments();
  auto sourcesBuilder = bodyBuilder.initSources(numSources);
  for (unsigned i = 0; i < numSources; ++i) {
    sourcesBuilder.set(i, ctx.getValueId(entryBlock.getArgument(i)));
  }

  // Set targets
  auto returnOp = llvm::cast<mlir::func::ReturnOp>(entryBlock.back());
  const auto numTargets = returnOp.getNumOperands();
  auto targetsBuilder = bodyBuilder.initTargets(numTargets);
  for (unsigned i = 0; i < numTargets; ++i) {
    targetsBuilder.set(i, ctx.getValueId(returnOp.getOperand(i)));
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
  auto functionBuilder = functionsBuilder[0];
  functionBuilder.setName(ctx.strings[function.getName().str()]);
  serializeFunction(functionBuilder, function, ctx);

  // Set metadata
  moduleBuilder.setTool("");
  moduleBuilder.setToolVersion("");

  return capnp::messageToFlatArray(message);
}
