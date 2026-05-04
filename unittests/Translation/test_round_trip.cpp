#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"
#include "jeff/Translation/Serialize.hpp"

#include <capnp/common.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <gtest/gtest.h>
#include <jeff.capnp.h>
#include <kj/common.h>
#include <kj/io.h>
#include <kj/string-tree.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>

#include <algorithm>
#include <filesystem>
#include <ostream>
#include <string>
#include <utility>
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

std::string readJeffFileToText(llvm::StringRef path) {
    llvm::sys::fs::file_t file = 0;
    if (llvm::sys::fs::openFileForRead(path, file)) {
        llvm::errs() << "Failed to open file: " << path << "\n";
        llvm::report_fatal_error("Could not open file");
    }

#ifdef _WIN32
    kj::AutoCloseHandle autoCloseHandle(file);
    kj::HandleInputStream input(std::move(autoCloseHandle));
#else
    kj::AutoCloseFd autoCloseFd(file);
    kj::FdInputStream input(std::move(autoCloseFd));
#endif

    capnp::MallocMessageBuilder message;
    capnp::readMessageCopy(input, message);
    const auto module = message.getRoot<jeff::Module>();
    return module.toString().flatten().cStr();
}

std::string moduleTextFromBuffer(const kj::ArrayPtr<capnp::word>& buffer) {
    capnp::FlatArrayMessageReader message(buffer);
    const auto module = message.getRoot<jeff::Module>();
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

    // Deserialize jeff module
    auto mlirModule = deserializeFromFile(&context, path.string());

    llvm::errs() << "Deserialized MLIR module:\n";
    mlirModule->print(llvm::errs());
    llvm::errs() << "\n\n";

    // Serialize MLIR module
    auto serialized = serialize(*mlirModule);

    // Compare textual representations
    const auto originalText = readJeffFileToText(path.string());
    const auto serializedText = moduleTextFromBuffer(serialized);

    llvm::errs() << "Original module:\n" << originalText << "\n\n";
    llvm::errs() << "Serialized module:\n" << serializedText << "\n\n";

    ASSERT_EQ(originalText, serializedText);
}

INSTANTIATE_TEST_SUITE_P(, RoundTripTest, ::testing::ValuesIn(getTestCases()));
