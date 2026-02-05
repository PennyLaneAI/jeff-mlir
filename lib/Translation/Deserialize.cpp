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
    jeff::QubitGate::Custom::Reader custom, ::capnp::Text::Reader name,
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader inputs,
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader outputs) {
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
      controls.size(), false, 1, name.cStr(), qubits.size(), 0);
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
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader inputs,
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader outputs) {
  llvm::SmallVector<mlir::Value> controls;
  if (inputs.size() >= 1) {
    for (size_t i = 1; i < inputs.size(); ++i) {
      controls.push_back(mlirValues[inputs[i]]);
    }
  }
  auto op =
      builder.create<OpType>(builder.getUnknownLoc(), mlirValues[inputs[0]],
                             controls, controls.size(), false, 1);
  mlirValues[outputs[0]] = op.getOutQubit();
  if (outputs.size() >= 1) {
    for (size_t i = 1; i < outputs.size(); ++i) {
      mlirValues[outputs[i]] = op.getOutCtrlQubits()[i - 1];
    }
  }
}

void convertWellKnown(
    mlir::OpBuilder& builder, llvm::DenseMap<int, mlir::Value>& mlirValues,
    jeff::WellKnownGate wellKnown,
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader inputs,
    ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader outputs) {
  if (wellKnown == jeff::WellKnownGate::H) {
    createOneTargetZeroParameter<mlir::jeff::HOp>(builder, mlirValues, inputs,
                                                  outputs);
    return;
  }
  if (wellKnown == jeff::WellKnownGate::X) {
    createOneTargetZeroParameter<mlir::jeff::XOp>(builder, mlirValues, inputs,
                                                  outputs);
    return;
  }
}

void parseOperations(
    mlir::OpBuilder& builder,
    ::capnp::List<jeff::Op, ::capnp::Kind::STRUCT>::Reader operations,
    llvm::DenseMap<int, mlir::Value>& mlirValues,
    ::capnp::List<::capnp::Text, ::capnp::Kind::BLOB>::Reader strings) {
  for (auto operation : operations) {
    auto instruction = operation.getInstruction();
    if (instruction.hasQubit()) {
      auto qubit = instruction.getQubit();
      if (qubit.isAlloc()) {
        convertAlloc(builder, mlirValues, operation.getOutputs());
      }
      if (qubit.hasGate()) {
        auto gate = qubit.getGate();
        if (gate.isCustom()) {
          auto custom = gate.getCustom();
          auto name = strings[custom.getName()];
          convertCustom(builder, mlirValues, custom, name,
                        operation.getInputs(), operation.getOutputs());
        }
        if (gate.isWellKnown()) {
          auto wellKnown = gate.getWellKnown();
          convertWellKnown(builder, mlirValues, wellKnown,
                           operation.getInputs(), operation.getOutputs());
        }
      }
    }
  }
}

} // namespace

mlir::OwningOpRef<mlir::ModuleOp> deserialize(mlir::MLIRContext* context,
                                              const std::string& path) {
  // Get Jeff module from file
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    llvm::report_fatal_error("Error: Could not open file");
  }
  ::capnp::StreamFdMessageReader message(fd);
  jeff::Module::Reader jeffModule = message.getRoot<jeff::Module>();

  // Get operations
  if (!jeffModule.hasFunctions()) {
    llvm::report_fatal_error("Error: No functions found in module");
  }
  auto functions = jeffModule.getFunctions();

  if (functions.size() != 1) {
    llvm::report_fatal_error("Error: Expected exactly one function in module");
  }
  auto function = functions[0];
  auto definition = function.getDefinition();

  if (!definition.hasBody()) {
    llvm::report_fatal_error("Error: Function definition has no body");
  }
  auto body = definition.getBody();

  if (!body.hasOperations()) {
    llvm::report_fatal_error("Error: Function body has no operations");
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

  parseOperations(builder, operations, mlirValues, strings);

  return mlirModule;
}
