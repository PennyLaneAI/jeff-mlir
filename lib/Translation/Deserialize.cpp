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
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <string>

namespace {

void convertAlloc(
    mlir::OpBuilder& builder, llvm::DenseMap<int, mlir::Value>& mlirValues,
    capnp::List<::uint32_t, capnp::Kind::PRIMITIVE>::Reader outputs) {
  auto allocOp =
      builder.create<mlir::jeff::QubitAllocOp>(builder.getUnknownLoc());
  mlirValues[outputs[0]] = allocOp.getResult();
}

void convertFree(
    mlir::OpBuilder& builder, llvm::DenseMap<int, mlir::Value>& mlirValues,
    capnp::List<::uint32_t, capnp::Kind::PRIMITIVE>::Reader inputs) {
  builder.create<mlir::jeff::QubitFreeOp>(builder.getUnknownLoc(),
                                          mlirValues[inputs[0]]);
}

void convertCustom(
    mlir::OpBuilder& builder, llvm::DenseMap<int, mlir::Value>& mlirValues,
    jeff::QubitGate::Reader gate, capnp::Text::Reader name,
    capnp::List<::uint32_t, capnp::Kind::PRIMITIVE>::Reader inputs,
    capnp::List<::uint32_t, capnp::Kind::PRIMITIVE>::Reader outputs) {
  const auto custom = gate.getCustom();
  const auto numTargets = custom.getNumQubits() - gate.getControlQubits();
  llvm::SmallVector<mlir::Value> targets;
  for (std::uint8_t i = 0; i < numTargets; ++i) {
    targets.push_back(mlirValues[inputs[i]]);
  }
  llvm::SmallVector<mlir::Value> controls;
  for (std::uint8_t i = numTargets; i < inputs.size(); ++i) {
    controls.push_back(mlirValues[inputs[i]]);
  }
  auto op = builder.create<mlir::jeff::CustomOp>(
      builder.getUnknownLoc(), targets, controls, mlir::ValueRange{},
      static_cast<uint8_t>(controls.size()), gate.getAdjoint(), gate.getPower(),
      name.cStr(), static_cast<uint8_t>(numTargets), custom.getNumParams());
  for (std::uint8_t i = 0; i < numTargets; ++i) {
    mlirValues[outputs[i]] = op.getOutTargetQubits()[i];
  }
  for (std::uint8_t i = numTargets; i < outputs.size(); ++i) {
    mlirValues[outputs[i]] = op.getOutCtrlQubits()[i - numTargets];
  }
}

template <typename OpType>
void createOneTargetZeroParameter(
    mlir::OpBuilder& builder, llvm::DenseMap<int, mlir::Value>& mlirValues,
    jeff::QubitGate::Reader gate,
    capnp::List<::uint32_t, capnp::Kind::PRIMITIVE>::Reader inputs,
    capnp::List<::uint32_t, capnp::Kind::PRIMITIVE>::Reader outputs) {
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
    capnp::List<::uint32_t, capnp::Kind::PRIMITIVE>::Reader inputs,
    capnp::List<::uint32_t, capnp::Kind::PRIMITIVE>::Reader outputs) {
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
    capnp::List<::uint32_t, capnp::Kind::PRIMITIVE>::Reader inputs,
    capnp::List<::uint32_t, capnp::Kind::PRIMITIVE>::Reader outputs) {
  auto op = builder.create<mlir::jeff::QubitMeasureOp>(builder.getUnknownLoc(),
                                                       mlirValues[inputs[0]]);
  mlirValues[outputs[0]] = op.getResult();
}

void convertQubit(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                  llvm::DenseMap<int, mlir::Value>& mlirValues,
                  capnp::List<capnp::Text, capnp::Kind::BLOB>::Reader strings) {
  const auto instruction = operation.getInstruction();
  const auto qubit = instruction.getQubit();
  if (qubit.isAlloc()) {
    convertAlloc(builder, mlirValues, operation.getOutputs());
  } else if (qubit.isFree()) {
    convertFree(builder, mlirValues, operation.getInputs());
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

void convertIntConst8(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                      llvm::DenseMap<int, mlir::Value>& mlirValues) {
  const auto instruction = operation.getInstruction();
  const auto intInstruction = instruction.getInt();
  const auto const8 = intInstruction.getConst8();
  auto intAttr = mlir::IntegerAttr::get(builder.getI8Type(), const8);
  auto op = builder.create<mlir::jeff::IntConst8Op>(
      builder.getUnknownLoc(), builder.getI8Type(), intAttr);
  mlirValues[operation.getOutputs()[0]] = op.getConstant();
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

void convertIntNot(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                   llvm::DenseMap<int, mlir::Value>& mlirValues) {
  auto op = builder.create<mlir::jeff::IntUnaryOp>(
      builder.getUnknownLoc(), mlirValues[operation.getInputs()[0]],
      mlir::jeff::IntUnaryOperation::_not);
  mlirValues[operation.getOutputs()[0]] = op.getB();
}

void convertIntAdd(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                   llvm::DenseMap<int, mlir::Value>& mlirValues) {
  auto op = builder.create<mlir::jeff::IntBinaryOp>(
      builder.getUnknownLoc(), mlirValues[operation.getInputs()[0]],
      mlirValues[operation.getInputs()[1]],
      mlir::jeff::IntBinaryOperation::_add);
  mlirValues[operation.getOutputs()[0]] = op.getC();
}

void convertIntShl(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                   llvm::DenseMap<int, mlir::Value>& mlirValues) {
  auto op = builder.create<mlir::jeff::IntBinaryOp>(
      builder.getUnknownLoc(), mlirValues[operation.getInputs()[0]],
      mlirValues[operation.getInputs()[1]],
      mlir::jeff::IntBinaryOperation::_shl);
  mlirValues[operation.getOutputs()[0]] = op.getC();
}

void convertInt(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                llvm::DenseMap<int, mlir::Value>& mlirValues) {
  const auto instruction = operation.getInstruction();
  const auto intInstruction = instruction.getInt();
  if (intInstruction.isConst8()) {
    convertIntConst8(builder, operation, mlirValues);
  } else if (intInstruction.isConst32()) {
    convertIntConst32(builder, operation, mlirValues);
  } else if (intInstruction.isNot()) {
    convertIntNot(builder, operation, mlirValues);
  } else if (intInstruction.isAdd()) {
    convertIntAdd(builder, operation, mlirValues);
  } else if (intInstruction.isShl()) {
    convertIntShl(builder, operation, mlirValues);
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

void convertIntArrayGetIndex(mlir::OpBuilder& builder,
                             jeff::Op::Reader operation,
                             llvm::DenseMap<int, mlir::Value>& mlirValues) {
  auto tensorType = mlirValues[operation.getInputs()[0]].getType();
  auto entryType =
      llvm::cast<mlir::RankedTensorType>(tensorType).getElementType();
  auto op = builder.create<mlir::jeff::IntArrayGetIndexOp>(
      builder.getUnknownLoc(), entryType, mlirValues[operation.getInputs()[0]],
      mlirValues[operation.getInputs()[1]]);
  mlirValues[operation.getOutputs()[0]] = op.getValue();
}

void convertIntArraySetIndex(mlir::OpBuilder& builder,
                             jeff::Op::Reader operation,
                             llvm::DenseMap<int, mlir::Value>& mlirValues) {
  auto tensorType = mlirValues[operation.getInputs()[0]].getType();
  auto op = builder.create<mlir::jeff::IntArraySetIndexOp>(
      builder.getUnknownLoc(), tensorType, mlirValues[operation.getInputs()[0]],
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
  } else if (intArray.isGetIndex()) {
    convertIntArrayGetIndex(builder, operation, mlirValues);
  } else if (intArray.isSetIndex()) {
    convertIntArraySetIndex(builder, operation, mlirValues);
  } else {
    llvm::errs() << "Cannot convert int array instruction "
                 << static_cast<int>(intArray.which()) << "\n";
    llvm::report_fatal_error("Unknown int array instruction");
  }
}

// Forward declaration
void convertOperations(
    mlir::OpBuilder& builder,
    capnp::List<jeff::Op, capnp::Kind::STRUCT>::Reader operations,
    llvm::DenseMap<int, mlir::Value>& mlirValues,
    llvm::DenseMap<int, mlir::func::FuncOp>& mlirFuncs,
    capnp::List<capnp::Text, capnp::Kind::BLOB>::Reader strings);

void convertSwitch(
    mlir::OpBuilder& builder, jeff::Op::Reader operation,
    llvm::DenseMap<int, mlir::Value>& mlirValues,
    llvm::DenseMap<int, mlir::func::FuncOp>& mlirFuncs,
    capnp::List<capnp::Text, capnp::Kind::BLOB>::Reader strings) {
  const auto instruction = operation.getInstruction();
  const auto scf = instruction.getScf();
  const auto _switch = scf.getSwitch();
  const auto branches = _switch.getBranches();

  llvm::SmallVector<mlir::Value> inValues;
  llvm::SmallVector<mlir::Type> outTypes;
  inValues.reserve(operation.getInputs().size() - 1);
  outTypes.reserve(operation.getInputs().size() - 1);
  for (size_t i = 1; i < operation.getInputs().size(); ++i) {
    inValues.push_back(mlirValues[operation.getInputs()[i]]);
    outTypes.push_back(mlirValues[operation.getInputs()[i]].getType());
  }

  auto op = builder.create<mlir::jeff::SwitchOp>(
      builder.getUnknownLoc(), outTypes, mlirValues[operation.getInputs()[0]],
      inValues, branches.size());

  if (_switch.hasDefault()) {
    const auto& _default = _switch.getDefault();
    for (size_t i = 0; i < _default.getSources().size(); ++i) {
      mlirValues[_default.getSources()[i]] = inValues[i];
    }
    auto& defaultRegion = op.getDefault();
    auto& defaultBlock = defaultRegion.emplaceBlock();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&defaultBlock);
    convertOperations(builder, _default.getOperations(), mlirValues, mlirFuncs,
                      strings);
    llvm::SmallVector<mlir::Value> outValues;
    outValues.reserve(_default.getTargets().size());
    for (size_t i = 0; i < _default.getTargets().size(); ++i) {
      outValues.push_back(mlirValues[_default.getTargets()[i]]);
    }
    builder.create<mlir::jeff::YieldOp>(builder.getUnknownLoc(), outValues);
  }

  for (size_t i = 0; i < branches.size(); ++i) {
    const auto& branch = branches[i];
    for (size_t j = 0; j < branch.getSources().size(); ++j) {
      mlirValues[branch.getSources()[j]] = inValues[j];
    }
    auto& branchRegion = op.getBranches()[i];
    auto& branchBlock = branchRegion.emplaceBlock();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&branchBlock);
    convertOperations(builder, branches[i].getOperations(), mlirValues,
                      mlirFuncs, strings);
    llvm::SmallVector<mlir::Value> outValues;
    outValues.reserve(branch.getTargets().size());
    for (size_t j = 0; j < branch.getTargets().size(); ++j) {
      outValues.push_back(mlirValues[branch.getTargets()[j]]);
    }
    builder.create<mlir::jeff::YieldOp>(builder.getUnknownLoc(), outValues);
  }

  llvm::SmallVector<mlir::Value> outValues;
  outValues.reserve(operation.getOutputs().size());
  for (size_t i = 0; i < operation.getOutputs().size(); ++i) {
    mlirValues[operation.getOutputs()[i]] = op.getResults()[i];
  }
}

void convertScf(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                llvm::DenseMap<int, mlir::Value>& mlirValues,
                llvm::DenseMap<int, mlir::func::FuncOp>& mlirFuncs,
                capnp::List<capnp::Text, capnp::Kind::BLOB>::Reader strings) {
  const auto instruction = operation.getInstruction();
  const auto scf = instruction.getScf();
  if (scf.isSwitch()) {
    convertSwitch(builder, operation, mlirValues, mlirFuncs, strings);
  } else {
    llvm::errs() << "Cannot convert scf instruction "
                 << static_cast<int>(scf.which()) << "\n";
    llvm::report_fatal_error("Unknown scf instruction");
  }
}

void convertFunc(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                 llvm::DenseMap<int, mlir::Value>& mlirValues,
                 llvm::DenseMap<int, mlir::func::FuncOp>& mlirFuncs,
                 capnp::List<capnp::Text, capnp::Kind::BLOB>::Reader strings) {
  const auto instruction = operation.getInstruction();
  const auto jeffFunc = instruction.getFunc();
  auto mlirFunc = mlirFuncs[jeffFunc.getFuncCall()];
  llvm::SmallVector<mlir::Value> inputs;
  inputs.reserve(operation.getInputs().size());
  for (const auto input : operation.getInputs()) {
    inputs.push_back(mlirValues[input]);
  }
  auto op = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(),
                                               mlirFunc, inputs);
  for (size_t i = 0; i < operation.getOutputs().size(); ++i) {
    mlirValues[operation.getOutputs()[i]] = op.getResult(i);
  }
}

void convertOperations(
    mlir::OpBuilder& builder,
    capnp::List<jeff::Op, capnp::Kind::STRUCT>::Reader operations,
    llvm::DenseMap<int, mlir::Value>& mlirValues,
    llvm::DenseMap<int, mlir::func::FuncOp>& mlirFuncs,
    capnp::List<capnp::Text, capnp::Kind::BLOB>::Reader strings) {
  for (auto operation : operations) {
    const auto instruction = operation.getInstruction();
    if (instruction.isQubit()) {
      convertQubit(builder, operation, mlirValues, strings);
    } else if (instruction.isInt()) {
      convertInt(builder, operation, mlirValues);
    } else if (instruction.isIntArray()) {
      convertIntArray(builder, operation, mlirValues);
    } else if (instruction.isScf()) {
      convertScf(builder, operation, mlirValues, mlirFuncs, strings);
    } else if (instruction.isFunc()) {
      convertFunc(builder, operation, mlirValues, mlirFuncs, strings);
    } else {
      llvm::errs() << "Cannot convert instruction "
                   << static_cast<int>(instruction.which()) << "\n";
      llvm::report_fatal_error("Unknown instruction");
    }
  }
}

void convertFunction(mlir::OpBuilder& builder, jeff::Function::Reader function,
                     jeff::Module::Reader jeffModule,
                     llvm::DenseMap<int, mlir::func::FuncOp>& mlirFuncs) {
  // Get strings
  const auto strings = jeffModule.getStrings();

  // Get entry point
  const auto entryPoint = jeffModule.getEntrypoint();

  // Get function definition
  const auto definition = function.getDefinition();

  // Get values
  const auto jeffValues = definition.getValues();

  llvm::DenseMap<int, mlir::Value> mlirValues;
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
    const auto type = jeffValues[source].getType();
    if (type.isInt()) {
      if (type.getInt() == 1) {
        sourceTypes.push_back(builder.getI1Type());
      } else if (type.getInt() == 8) {
        sourceTypes.push_back(builder.getI8Type());
      } else {
        llvm::errs() << "Cannot convert source int type "
                     << static_cast<int>(type.getInt()) << "\n";
        llvm::report_fatal_error("Unknown source int type");
      }
    } else {
      llvm::errs() << "Cannot convert source type "
                   << static_cast<int>(type.which()) << "\n";
      llvm::report_fatal_error("Unknown source type");
    }
  }

  // Get targets
  const auto targets = body.getTargets();

  // Get target types
  llvm::SmallVector<mlir::Type> targetTypes;
  targetTypes.reserve(targets.size());
  for (auto target : targets) {
    if (target >= jeffValues.size()) {
      llvm::outs() << "WARNING: Target index " << target << " out of bounds\n";
      target = jeffValues.size() - 1;
    }
    const auto type = jeffValues[target].getType();
    if (type.isInt()) {
      if (type.getInt() == 1) {
        targetTypes.push_back(builder.getI1Type());
      } else if (type.getInt() == 8) {
        targetTypes.push_back(builder.getI8Type());
      } else if (type.getInt() == 32) {
        targetTypes.push_back(builder.getI32Type());
      } else {
        llvm::errs() << "Cannot convert target int type "
                     << static_cast<int>(type.getInt()) << "\n";
        llvm::report_fatal_error("Unknown target int type");
      }
    } else {
      llvm::errs() << "Cannot convert target type "
                   << static_cast<int>(type.which()) << "\n";
      llvm::report_fatal_error("Unknown target type");
    }
  }

  // Create function
  const auto funcName = strings[function.getName()].cStr();
  auto funcType = builder.getFunctionType(sourceTypes, targetTypes);
  auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                 funcName, funcType);
  mlirFuncs[function.getName()] = func;

  // Add attributes if the function is the entry point
  if (function.getName() == entryPoint) {
    llvm::SmallVector<mlir::Attribute> attributes;
    attributes.emplace_back(builder.getStringAttr("entry_point"));
    const auto jeffVersion = std::to_string(jeffModule.getVersion());
    attributes.emplace_back(builder.getStrArrayAttr({"version", jeffVersion}));
    func->setAttr("passthrough", builder.getArrayAttr(attributes));
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  auto& entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Add sources to mlirValues
  for (auto i = 0; i < sources.size(); ++i) {
    mlirValues[sources[i]] = entryBlock.getArgument(i);
  }

  convertOperations(builder, operations, mlirValues, mlirFuncs, strings);

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
  const auto fd = open(path.c_str(), O_RDONLY);
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

  if (!jeffModule.hasFunctions()) {
    llvm::report_fatal_error("No functions found in module");
  }
  const auto functions = jeffModule.getFunctions();

  llvm::DenseMap<int, mlir::func::FuncOp> mlirFuncs;
  mlirFuncs.reserve(functions.size());

  for (const auto function : functions) {
    convertFunction(builder, function, jeffModule, mlirFuncs);
  }

  return mlirModule;
}
