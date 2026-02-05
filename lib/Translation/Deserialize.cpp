#include "jeff/IR/JeffDialect.h"
#include "jeff/IR/JeffOps.h"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <fcntl.h>
#include <iostream>
#include <jeff.capnp.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <unistd.h>

namespace {

void convertAlloc(
    mlir::OpBuilder& builder, llvm::DenseMap<int, mlir::Value>& mlirValues,
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader outputs) {
  auto allocOp =
      builder.create<mlir::jeff::QubitAllocOp>(builder.getUnknownLoc());
  mlirValues[outputs[0]] = allocOp.getResult();
}

void convertCustom(
    mlir::OpBuilder& builder, llvm::DenseMap<int, mlir::Value>& mlirValues,
    jeff::QubitGate::Reader gate, ::capnp::Text::Reader name,
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader inputs,
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader outputs) {
  const auto custom = gate.getCustom();
  const auto numQubits = custom.getNumQubits();
  llvm::SmallVector<mlir::Value> qubits;
  for (std::uint8_t i = 0; i < numQubits; ++i) {
    qubits.push_back(mlirValues[inputs[i]]);
  }
  llvm::SmallVector<mlir::Value> controls;
  for (std::uint8_t i = numQubits; i < inputs.size(); ++i) {
    controls.push_back(mlirValues[inputs[i]]);
  }
  auto op = builder.create<mlir::jeff::CustomOp>(
      builder.getUnknownLoc(), qubits, controls, mlir::ValueRange{},
      static_cast<uint8_t>(controls.size()), gate.getAdjoint(), gate.getPower(),
      name.cStr(), static_cast<uint8_t>(qubits.size()), custom.getNumParams());
  for (std::uint8_t i = 0; i < numQubits; ++i) {
    mlirValues[outputs[i]] = op.getOutQubits()[i];
  }
  for (std::uint8_t i = numQubits; i < outputs.size(); ++i) {
    mlirValues[outputs[i]] = op.getOutCtrlQubits()[i - numQubits];
  }
}

template <typename OpType>
void createOneTargetZeroParameter(
    mlir::OpBuilder& builder, llvm::DenseMap<int, mlir::Value>& mlirValues,
    jeff::QubitGate::Reader gate,
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader inputs,
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader outputs) {
  const auto wellKnown = gate.getWellKnown();
  llvm::SmallVector<mlir::Value> controls;
  if (inputs.size() >= 1) {
    for (size_t i = 1; i < inputs.size(); ++i) {
      controls.push_back(mlirValues[inputs[i]]);
    }
  }
  auto op =
      builder.create<OpType>(builder.getUnknownLoc(), mlirValues[inputs[0]],
                             controls, static_cast<uint8_t>(controls.size()),
                             gate.getAdjoint(), gate.getPower());
  mlirValues[outputs[0]] = op.getOutQubit();
  if (outputs.size() >= 1) {
    for (size_t i = 1; i < outputs.size(); ++i) {
      mlirValues[outputs[i]] = op.getOutCtrlQubits()[i - 1];
    }
  }
}

void convertWellKnown(
    mlir::OpBuilder& builder, llvm::DenseMap<int, mlir::Value>& mlirValues,
    jeff::QubitGate::Reader gate,
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader inputs,
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader outputs) {
  const auto wellKnown = gate.getWellKnown();
  switch (wellKnown) {
  case jeff::WellKnownGate::H:
    createOneTargetZeroParameter<mlir::jeff::HOp>(builder, mlirValues, gate,
                                                  inputs, outputs);
    break;
  case jeff::WellKnownGate::X:
    createOneTargetZeroParameter<mlir::jeff::XOp>(builder, mlirValues, gate,
                                                  inputs, outputs);
    break;
  default:
    llvm::errs() << "Cannot convert well-known gate "
                 << static_cast<int>(wellKnown) << "\n";
    llvm::report_fatal_error("Unknown well-known gate");
  }
}

void convertMeasure(
    mlir::OpBuilder& builder, llvm::DenseMap<int, mlir::Value>& mlirValues,
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader inputs,
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader outputs) {
  auto op = builder.create<mlir::jeff::QubitMeasureOp>(builder.getUnknownLoc(),
                                                       mlirValues[inputs[0]]);
  mlirValues[outputs[0]] = op.getResult();
}

void convertQubit(
    mlir::OpBuilder& builder, jeff::Op::Reader operation,
    llvm::DenseMap<int, mlir::Value>& mlirValues,
    ::capnp::List<::capnp::Text, ::capnp::Kind::BLOB>::Reader strings) {
  const auto instruction = operation.getInstruction();
  const auto qubit = instruction.getQubit();
  if (qubit.isAlloc()) {
    convertAlloc(builder, mlirValues, operation.getOutputs());
  } else if (qubit.isGate()) {
    const auto gate = qubit.getGate();
    if (gate.isCustom()) {
      const auto name = strings[gate.getCustom().getName()];
      convertCustom(builder, mlirValues, gate, name, operation.getInputs(),
                    operation.getOutputs());
    } else if (gate.isWellKnown()) {
      convertWellKnown(builder, mlirValues, gate, operation.getInputs(),
                       operation.getOutputs());
    }
  } else if (qubit.isMeasure()) {
    convertMeasure(builder, mlirValues, operation.getInputs(),
                   operation.getOutputs());
  } else {
    llvm::errs() << "Cannot convert qubit instruction "
                 << static_cast<int>(qubit.which()) << "\n";
    llvm::report_fatal_error("Unknown qubit instruction");
  }
}

void convertIntConst32(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                       llvm::DenseMap<int, mlir::Value>& mlirValues) {
  const auto instruction = operation.getInstruction();
  const auto intInstruction = instruction.getInt();
  const auto const32 = intInstruction.getConst32();
  auto intAttr = mlir::IntegerAttr::get(builder.getI32Type(), const32);
  auto op = builder.create<mlir::jeff::IntConst32Op>(
      builder.getUnknownLoc(), builder.getI32Type(), intAttr);
  mlirValues[operation.getOutputs()[0]] = op.getConstant();
}

void convertInt(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                llvm::DenseMap<int, mlir::Value>& mlirValues) {
  const auto instruction = operation.getInstruction();
  const auto intInstruction = instruction.getInt();
  if (intInstruction.isConst32()) {
    convertIntConst32(builder, operation, mlirValues);
  } else {
    llvm::errs() << "Cannot convert int instruction "
                 << static_cast<int>(intInstruction.which()) << "\n";
    llvm::report_fatal_error("Unknown int instruction");
  }
}

void convertIntArrayConst8(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                           llvm::DenseMap<int, mlir::Value>& mlirValues) {
  const auto instruction = operation.getInstruction();
  const auto intArray = instruction.getIntArray();
  const auto const8 = intArray.getConst8();
  llvm::SmallVector<int8_t> inArray;
  inArray.reserve(const8.size());
  for (auto value : const8) {
    inArray.push_back(static_cast<int8_t>(value));
  }
  auto inArrayAttr = mlir::DenseI8ArrayAttr::get(builder.getContext(), inArray);
  auto tensorType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(inArray.size())}, builder.getI8Type());
  auto op = builder.create<mlir::jeff::IntArrayConst8Op>(
      builder.getUnknownLoc(), tensorType, inArrayAttr);
  mlirValues[operation.getOutputs()[0]] = op.getOutArray();
}

void convertIntArraySetIndex(mlir::OpBuilder& builder,
                             jeff::Op::Reader operation,
                             llvm::DenseMap<int, mlir::Value>& mlirValues) {
  auto op = builder.create<mlir::jeff::IntArraySetIndexOp>(
      builder.getUnknownLoc(), mlirValues[operation.getInputs()[0]].getType(),
      mlirValues[operation.getInputs()[0]],
      mlirValues[operation.getInputs()[1]],
      mlirValues[operation.getInputs()[2]]);
  mlirValues[operation.getOutputs()[0]] = op.getOutArray();
}

void convertIntArray(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                     llvm::DenseMap<int, mlir::Value>& mlirValues) {
  const auto instruction = operation.getInstruction();
  const auto intArray = instruction.getIntArray();
  if (intArray.isConst8()) {
    convertIntArrayConst8(builder, operation, mlirValues);
  } else if (intArray.isSetIndex()) {
    convertIntArraySetIndex(builder, operation, mlirValues);
  } else {
    llvm::errs() << "Cannot convert int array instruction "
                 << static_cast<int>(intArray.which()) << "\n";
    llvm::report_fatal_error("Unknown int array instruction");
  }
}

void convertOperations(
    mlir::OpBuilder& builder,
    ::capnp::List<jeff::Op, ::capnp::Kind::STRUCT>::Reader operations,
    llvm::DenseMap<int, mlir::Value>& mlirValues,
    ::capnp::List<::capnp::Text, ::capnp::Kind::BLOB>::Reader strings) {
  for (auto operation : operations) {
    const auto instruction = operation.getInstruction();
    if (instruction.isQubit()) {
      convertQubit(builder, operation, mlirValues, strings);
    } else if (instruction.isInt()) {
      convertInt(builder, operation, mlirValues);
    } else if (instruction.isIntArray()) {
      convertIntArray(builder, operation, mlirValues);
    } else {
      llvm::errs() << "Cannot convert instruction "
                   << static_cast<int>(instruction.which()) << "\n";
      llvm::report_fatal_error("Unknown instruction");
    }
  }
}

} // namespace

mlir::OwningOpRef<mlir::ModuleOp> deserialize(mlir::MLIRContext* context,
                                              const std::string& path) {
  // Get Jeff module from file
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    llvm::report_fatal_error("Could not open file");
  }
  ::capnp::StreamFdMessageReader message(fd);
  jeff::Module::Reader jeffModule = message.getRoot<jeff::Module>();

  // Get operations
  if (!jeffModule.hasFunctions()) {
    llvm::report_fatal_error("No functions found in module");
  }
  auto functions = jeffModule.getFunctions();

  if (functions.size() != 1) {
    llvm::report_fatal_error("Expected exactly one function in module");
  }
  auto function = functions[0];
  auto definition = function.getDefinition();

  if (!definition.hasBody()) {
    llvm::report_fatal_error("Function definition has no body");
  }
  auto body = definition.getBody();

  if (!body.hasOperations()) {
    llvm::report_fatal_error("Function body has no operations");
  }
  auto operations = body.getOperations();

  // Get values
  auto jeffValues = definition.getValues();

  llvm::DenseMap<int, mlir::Value> mlirValues;
  mlirValues.reserve(jeffValues.size());

  // Get strings
  auto strings = jeffModule.getStrings();

  // Create MLIR builder
  mlir::OpBuilder builder(context);
  auto loc = builder.getUnknownLoc();

  // Create MLIR module
  auto mlirModule = builder.create<mlir::ModuleOp>(loc);
  builder.setInsertionPointToStart(mlirModule.getBody());

  convertOperations(builder, operations, mlirValues, strings);

  return mlirModule;
}
