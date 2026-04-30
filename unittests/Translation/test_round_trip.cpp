#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"
#include "jeff/Translation/Serialize.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <gtest/gtest.h>
#include <jeff.capnp.h>
#include <kj/common.h>
#include <kj/io.h>
#include <kj/string-tree.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct RoundTripTestCase {
    std::string filename;
};

std::ostream& operator<<(std::ostream& os, const RoundTripTestCase& testCase) {
    return os << testCase.filename;
}

class RoundTripTest : public ::testing::Test,
                      public ::testing::WithParamInterface<RoundTripTestCase> {};

llvm::SmallVector<uint8_t> readJeffFile(llvm::StringRef path) {
    auto file = llvm::MemoryBuffer::getFile(path);
    if (!file) {
        llvm::errs() << "Failed to open file: " << path << "\n";
        llvm::report_fatal_error("Could not open file");
    }

    auto bytes = file.get()->getBuffer();
    return {reinterpret_cast<const uint8_t*>(bytes.begin()),
            reinterpret_cast<const uint8_t*>(bytes.end())};
}

std::string moduleTextFromBytes(llvm::ArrayRef<uint8_t> data) {
    kj::ArrayPtr<const kj::byte> bytes(reinterpret_cast<const kj::byte*>(data.data()), data.size());
    kj::ArrayInputStream input(bytes);

    capnp::MallocMessageBuilder message;
    capnp::readMessageCopy(input, message);
    auto module = message.getRoot<jeff::Module>();
    return module.toString().flatten().cStr();
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
        cases.push_back({entry.path().filename().string()});
    }
    std::sort(cases.begin(), cases.end(),
              [](const auto& a, const auto& b) { return a.filename < b.filename; });
    return cases;
}

} // namespace

TEST_P(RoundTripTest, RoundTrip) {
    const auto& testCase = GetParam();

    if (testCase.filename.rfind("skip_", 0) == 0) {
        GTEST_SKIP();
    }

    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::jeff::JeffDialect>();

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    const fs::path inputsDir = TEST_INPUTS_DIR;
    const auto& path = inputsDir / testCase.filename;

    // Load original jeff module
    auto original = readJeffFile(path.string());

    // Deserialize jeff module
    auto mlirModule = deserializeFromFile(&context, path.string());

    llvm::errs() << "Deserialized MLIR module:\n";
    mlirModule->print(llvm::errs());
    llvm::errs() << "\n\n";

    // Serialize MLIR module
    auto serialized = serialize(*mlirModule);

    // Compare textual representations
    auto originalText = moduleTextFromBytes(original);
    auto serializedText = moduleTextFromBytes(serialized);

    llvm::errs() << "Original module:\n" << originalText << "\n\n";
    llvm::errs() << "Serialized module:\n" << serializedText << "\n\n";

    ASSERT_EQ(originalText, serializedText);
}

INSTANTIATE_TEST_SUITE_P(, RoundTripTest, ::testing::ValuesIn(getTestCases()));
