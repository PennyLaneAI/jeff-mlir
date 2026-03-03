#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"
#include "jeff/Translation/Serialize.hpp"

#include <capnp/common.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <fcntl.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <jeff.capnp.h>
#include <kj/array.h>
#include <kj/string-tree.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>
#include <ostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct RoundTripTestCase {
  std::string fileName;
};

std::ostream& operator<<(std::ostream& os, const RoundTripTestCase& testCase) {
  return os << testCase.fileName;
}

class RoundTripTest : public ::testing::Test,
                      public ::testing::WithParamInterface<RoundTripTestCase> {
};

kj::Array<capnp::word> readJeffFile(const std::string& path) {
  const auto fd = open(path.c_str(), O_RDONLY, 0);
  if (fd < 0) {
    llvm::errs() << "Failed to open file: " << path << "\n";
    llvm::report_fatal_error("Could not open file");
  }
  capnp::MallocMessageBuilder message;
  capnp::readMessageCopyFromFd(fd, message);
  return capnp::messageToFlatArray(message);
}

std::vector<RoundTripTestCase> getTestCases() {
  std::vector<RoundTripTestCase> cases;
  for (const auto& entry : fs::directory_iterator(TEST_INPUTS_DIR)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    if (entry.path().extension() != ".jeff") {
      continue;
    }
    const auto filename = entry.path().filename().string();
    if (filename.rfind("skip_", 0) == 0) {
      continue;
    }
    cases.push_back({filename});
  }
  std::sort(cases.begin(), cases.end(), [](const auto& a, const auto& b) {
    return a.fileName < b.fileName;
  });
  return cases;
}

} // namespace

TEST_P(RoundTripTest, RoundTrip) {
  const auto& testCase = GetParam();

  mlir::DialectRegistry registry;
  registry.insert<mlir::jeff::JeffDialect, mlir::func::FuncDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  const fs::path inputsDir = TEST_INPUTS_DIR;
  const auto& path = inputsDir / testCase.fileName;

  auto original = readJeffFile(path.string());

  // Deserialize and serialize again
  auto mlirModule = deserialize(&context, path.string());
  auto serialized = serializeToArray(*mlirModule);

  llvm::outs() << "Deserialized MLIR module:\n";
  mlirModule->print(llvm::outs());
  llvm::outs() << "\n\n";

  // Compare textual representations
  capnp::FlatArrayMessageReader originalMessage(original);
  auto originalModule = originalMessage.getRoot<jeff::Module>();
  auto originalText = originalModule.toString().flatten();

  capnp::FlatArrayMessageReader serializedMessage(serialized);
  auto serializedModule = serializedMessage.getRoot<jeff::Module>();
  auto serializedText = serializedModule.toString().flatten();

  llvm::errs() << "Original module:\n" << originalText.cStr() << "\n\n";
  llvm::errs() << "Serialized module:\n" << serializedText.cStr() << "\n\n";

  ASSERT_EQ(originalText, serializedText);
}

INSTANTIATE_TEST_SUITE_P(, RoundTripTest, ::testing::ValuesIn(getTestCases()));
