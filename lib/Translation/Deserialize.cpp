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

struct DeserializationData {
  llvm::DenseMap<int, mlir::Value>& mlirValues;
  llvm::DenseMap<int, mlir::func::FuncOp>& mlirFuncs;
  capnp::List<capnp::Text, capnp::Kind::BLOB>::Reader& strings;
};

void convertAlloc(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                  DeserializationData& data) {
  auto allocOp =
      builder.create<mlir::jeff::QubitAllocOp>(builder.getUnknownLoc());
  data.mlirValues[operation.getOutputs()[0]] = allocOp.getResult();
}

void convertFree(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                 DeserializationData& data) {
  builder.create<mlir::jeff::QubitFreeOp>(
      builder.getUnknownLoc(), data.mlirValues[operation.getInputs()[0]]);
}

void convertCustom(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                   DeserializationData& data) {
  auto& mlirValues = data.mlirValues;
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  const auto gate = operation.getInstruction().getQubit().getGate();
  const auto custom = gate.getCustom();
  const auto name = data.strings[custom.getName()].cStr();
  const auto numTargets = custom.getNumQubits();
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
      name, static_cast<uint8_t>(numTargets), custom.getNumParams());
  for (std::uint8_t i = 0; i < numTargets; ++i) {
    mlirValues[outputs[i]] = op.getOutTargetQubits()[i];
  }
  for (std::uint8_t i = numTargets; i < outputs.size(); ++i) {
    mlirValues[outputs[i]] = op.getOutCtrlQubits()[i - numTargets];
  }
}

template <typename OpType>
void createOneTargetZeroParameter(mlir::OpBuilder& builder,
                                  jeff::Op::Reader operation,
                                  DeserializationData& data) {
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  const auto gate = operation.getInstruction().getQubit().getGate();
  auto& mlirValues = data.mlirValues;
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

void convertWellKnown(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                      DeserializationData& data) {
  const auto wellKnown =
      operation.getInstruction().getQubit().getGate().getWellKnown();
  switch (wellKnown) {
  case jeff::WellKnownGate::H:
    createOneTargetZeroParameter<mlir::jeff::HOp>(builder, operation, data);
    break;
  case jeff::WellKnownGate::X:
    createOneTargetZeroParameter<mlir::jeff::XOp>(builder, operation, data);
    break;
  default:
    llvm::errs() << "Cannot convert well-known gate "
                 << static_cast<int>(wellKnown) << "\n";
    llvm::report_fatal_error("Unknown well-known gate");
  }
}

void convertMeasure(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                    DeserializationData& data) {
  auto& mlirValues = data.mlirValues;
  auto op = builder.create<mlir::jeff::QubitMeasureOp>(
      builder.getUnknownLoc(), mlirValues[operation.getInputs()[0]]);
  mlirValues[operation.getOutputs()[0]] = op.getResult();
}

void convertQubit(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                  DeserializationData& data) {
  const auto instruction = operation.getInstruction();
  const auto qubit = instruction.getQubit();
  if (qubit.isAlloc()) {
    convertAlloc(builder, operation, data);
  } else if (qubit.isFree()) {
    convertFree(builder, operation, data);
  } else if (qubit.isGate()) {
    const auto gate = qubit.getGate();
    if (gate.isCustom()) {
      convertCustom(builder, operation, data);
    } else if (gate.isWellKnown()) {
      convertWellKnown(builder, operation, data);
    }
  } else if (qubit.isMeasure()) {
    convertMeasure(builder, operation, data);
  } else {
    llvm::errs() << "Cannot convert qubit instruction "
                 << static_cast<int>(qubit.which()) << "\n";
    llvm::report_fatal_error("Unknown qubit instruction");
  }
}

template <typename ConstOp, typename IntType>
void convertIntConstOp(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                       IntType value, mlir::Type type,
                       DeserializationData& data) {
  auto& mlirValues = data.mlirValues;
  auto intAttr = mlir::IntegerAttr::get(type, value);
  auto op = builder.create<ConstOp>(builder.getUnknownLoc(), type, intAttr);
  mlirValues[operation.getOutputs()[0]] = op.getConstant();
}

void convertIntUnaryOp(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                       mlir::jeff::IntUnaryOperation unaryOperation,
                       DeserializationData& data) {
  auto& mlirValues = data.mlirValues;
  auto op = builder.create<mlir::jeff::IntUnaryOp>(
      builder.getUnknownLoc(), mlirValues[operation.getInputs()[0]],
      unaryOperation);
  mlirValues[operation.getOutputs()[0]] = op.getB();
}

void convertIntBinaryOp(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                        mlir::jeff::IntBinaryOperation binaryOperation,
                        DeserializationData& data) {
  auto& mlirValues = data.mlirValues;
  const auto inputs = operation.getInputs();
  auto op = builder.create<mlir::jeff::IntBinaryOp>(
      builder.getUnknownLoc(), mlirValues[inputs[0]], mlirValues[inputs[1]],
      binaryOperation);
  mlirValues[operation.getOutputs()[0]] = op.getC();
}

void convertInt(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                DeserializationData& data) {
  const auto intInstruction = operation.getInstruction().getInt();
  if (intInstruction.isConst8()) {
    convertIntConstOp<mlir::jeff::IntConst8Op, uint8_t>(
        builder, operation, intInstruction.getConst8(), builder.getI8Type(),
        data);
  } else if (intInstruction.isConst32()) {
    convertIntConstOp<mlir::jeff::IntConst32Op, uint32_t>(
        builder, operation, intInstruction.getConst32(), builder.getI32Type(),
        data);
  } else if (intInstruction.isNot()) {
    convertIntUnaryOp(builder, operation, mlir::jeff::IntUnaryOperation::_not,
                      data);
  } else if (intInstruction.isAdd()) {
    convertIntBinaryOp(builder, operation, mlir::jeff::IntBinaryOperation::_add,
                       data);
  } else if (intInstruction.isShl()) {
    convertIntBinaryOp(builder, operation, mlir::jeff::IntBinaryOperation::_shl,
                       data);
  } else {
    llvm::errs() << "Cannot convert int instruction "
                 << static_cast<int>(intInstruction.which()) << "\n";
    llvm::report_fatal_error("Unknown int instruction");
  }
}

void convertIntArrayConst8(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                           DeserializationData& data) {
  const auto const8 = operation.getInstruction().getIntArray().getConst8();
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
  data.mlirValues[operation.getOutputs()[0]] = op.getOutArray();
}

void convertIntArrayGetIndex(mlir::OpBuilder& builder,
                             jeff::Op::Reader operation,
                             DeserializationData& data) {
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto& mlirValues = data.mlirValues;
  auto tensorType = mlirValues[inputs[0]].getType();
  auto entryType =
      llvm::cast<mlir::RankedTensorType>(tensorType).getElementType();
  auto op = builder.create<mlir::jeff::IntArrayGetIndexOp>(
      builder.getUnknownLoc(), entryType, mlirValues[inputs[0]],
      mlirValues[inputs[1]]);
  mlirValues[outputs[0]] = op.getValue();
}

void convertIntArraySetIndex(mlir::OpBuilder& builder,
                             jeff::Op::Reader operation,
                             DeserializationData& data) {
  const auto inputs = operation.getInputs();
  const auto outputs = operation.getOutputs();
  auto& mlirValues = data.mlirValues;
  auto tensorType = mlirValues[inputs[0]].getType();
  auto op = builder.create<mlir::jeff::IntArraySetIndexOp>(
      builder.getUnknownLoc(), tensorType, mlirValues[inputs[0]],
      mlirValues[inputs[1]], mlirValues[inputs[2]]);
  mlirValues[outputs[0]] = op.getOutArray();
}

void convertIntArray(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                     DeserializationData& data) {
  const auto intArray = operation.getInstruction().getIntArray();
  if (intArray.isConst8()) {
    convertIntArrayConst8(builder, operation, data);
  } else if (intArray.isGetIndex()) {
    convertIntArrayGetIndex(builder, operation, data);
  } else if (intArray.isSetIndex()) {
    convertIntArraySetIndex(builder, operation, data);
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
    DeserializationData& data);

void convertSwitch(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                   DeserializationData& data) {
  auto& mlirValues = data.mlirValues;
  const auto instruction = operation.getInstruction();
  const auto scf = instruction.getScf();
  const auto switch_ = scf.getSwitch();
  const auto branches = switch_.getBranches();

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

  if (switch_.hasDefault()) {
    const auto& default_ = switch_.getDefault();
    for (size_t i = 0; i < default_.getSources().size(); ++i) {
      mlirValues[default_.getSources()[i]] = inValues[i];
    }
    auto& defaultRegion = op.getDefault();
    auto& defaultBlock = defaultRegion.emplaceBlock();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&defaultBlock);
    convertOperations(builder, default_.getOperations(), data);
    llvm::SmallVector<mlir::Value> outValues;
    outValues.reserve(default_.getTargets().size());
    for (size_t i = 0; i < default_.getTargets().size(); ++i) {
      outValues.push_back(mlirValues[default_.getTargets()[i]]);
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
    convertOperations(builder, branches[i].getOperations(), data);
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
                DeserializationData& data) {
  const auto instruction = operation.getInstruction();
  const auto scf = instruction.getScf();
  if (scf.isSwitch()) {
    convertSwitch(builder, operation, data);
  } else {
    llvm::errs() << "Cannot convert scf instruction "
                 << static_cast<int>(scf.which()) << "\n";
    llvm::report_fatal_error("Unknown scf instruction");
  }
}

void convertFunc(mlir::OpBuilder& builder, jeff::Op::Reader operation,
                 DeserializationData& data) {
  auto& mlirValues = data.mlirValues;
  const auto instruction = operation.getInstruction();
  const auto jeffFunc = instruction.getFunc();
  auto mlirFunc = data.mlirFuncs[jeffFunc.getFuncCall()];
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
    DeserializationData& data) {
  for (auto operation : operations) {
    const auto instruction = operation.getInstruction();
    if (instruction.isQubit()) {
      convertQubit(builder, operation, data);
    } else if (instruction.isInt()) {
      convertInt(builder, operation, data);
    } else if (instruction.isIntArray()) {
      convertIntArray(builder, operation, data);
    } else if (instruction.isScf()) {
      convertScf(builder, operation, data);
    } else if (instruction.isFunc()) {
      convertFunc(builder, operation, data);
    } else {
      llvm::errs() << "Cannot convert instruction "
                   << static_cast<int>(instruction.which()) << "\n";
      llvm::report_fatal_error("Unknown instruction");
    }
  }
}

mlir::Type convertType(mlir::OpBuilder& builder, jeff::Type::Reader type) {
  if (type.isInt()) {
    if (type.getInt() == 1) {
      return builder.getI1Type();
    } else if (type.getInt() == 8) {
      return builder.getI8Type();
    } else if (type.getInt() == 32) {
      return builder.getI32Type();
    } else {
      llvm::errs() << "Cannot convert int type "
                   << static_cast<int>(type.getInt()) << "\n";
      llvm::report_fatal_error("Unknown int type");
    }
  } else {
    llvm::errs() << "Cannot convert type " << static_cast<int>(type.which())
                 << "\n";
    llvm::report_fatal_error("Unknown type");
  }
}

void convertFunction(mlir::OpBuilder& builder, jeff::Function::Reader function,
                     jeff::Module::Reader jeffModule,
                     llvm::DenseMap<int, mlir::func::FuncOp>& mlirFuncs) {
  // Get strings
  auto strings = jeffModule.getStrings();

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
    const auto jeffType = jeffValues[source].getType();
    sourceTypes.push_back(convertType(builder, jeffType));
  }

  // Get targets
  const auto targets = body.getTargets();

  // Get target types
  llvm::SmallVector<mlir::Type> targetTypes;
  targetTypes.reserve(targets.size());
  for (auto target : targets) {
    const auto jeffType = jeffValues[target].getType();
    targetTypes.push_back(convertType(builder, jeffType));
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

  auto data = DeserializationData{mlirValues, mlirFuncs, strings};
  convertOperations(builder, operations, data);

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
