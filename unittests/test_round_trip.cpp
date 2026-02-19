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
  auto serialized = serialize(*mlirModule);

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

INSTANTIATE_TEST_SUITE_P(
    , RoundTripTest,
    ::testing::Values(
        // Qubit operations
        RoundTripTestCase{"unit_qubit_alloc.jeff"},
        RoundTripTestCase{"unit_qubit_free_zero.jeff"},
        RoundTripTestCase{"unit_qubit_free.jeff"},
        RoundTripTestCase{"unit_qubit_measure_nd.jeff"},
        RoundTripTestCase{"unit_qubit_measure.jeff"},
        RoundTripTestCase{"unit_qubit_reset.jeff"},
        // Gate operations
        RoundTripTestCase{"unit_gate_cgphase.jeff"},
        RoundTripTestCase{"unit_gate_cr1.jeff"},
        RoundTripTestCase{"unit_gate_crx.jeff"},
        RoundTripTestCase{"unit_gate_cry.jeff"},
        RoundTripTestCase{"unit_gate_crz.jeff"},
        RoundTripTestCase{"unit_gate_cswap.jeff"},
        RoundTripTestCase{"unit_gate_cu.jeff"},
        RoundTripTestCase{"unit_gate_custom_1.jeff"},
        RoundTripTestCase{"unit_gate_custom_2.jeff"},
        RoundTripTestCase{"unit_gate_gphase.jeff"},
        RoundTripTestCase{"unit_gate_h.jeff"},
        RoundTripTestCase{"unit_gate_i.jeff"},
        RoundTripTestCase{"unit_gate_mch.jeff"},
        RoundTripTestCase{"unit_gate_mci.jeff"},
        RoundTripTestCase{"unit_gate_mcs.jeff"},
        RoundTripTestCase{"unit_gate_mct.jeff"},
        RoundTripTestCase{"unit_gate_mcx.jeff"},
        RoundTripTestCase{"unit_gate_mcy.jeff"},
        RoundTripTestCase{"unit_gate_mcz.jeff"},
        RoundTripTestCase{"unit_gate_ppr_crxy.jeff"},
        RoundTripTestCase{"unit_gate_ppr_rxx.jeff"},
        RoundTripTestCase{"unit_gate_r1.jeff"},
        RoundTripTestCase{"unit_gate_rx.jeff"},
        RoundTripTestCase{"unit_gate_ry.jeff"},
        RoundTripTestCase{"unit_gate_rz.jeff"},
        RoundTripTestCase{"unit_gate_s.jeff"},
        RoundTripTestCase{"unit_gate_swap.jeff"},
        RoundTripTestCase{"unit_gate_t.jeff"},
        RoundTripTestCase{"unit_gate_u.jeff"},
        RoundTripTestCase{"unit_gate_x.jeff"},
        RoundTripTestCase{"unit_gate_y.jeff"},
        RoundTripTestCase{"unit_gate_z.jeff"},
        // Qureg operations
        RoundTripTestCase{"unit_qureg_alloc.jeff"},
        RoundTripTestCase{"unit_qureg_create.jeff"},
        RoundTripTestCase{"unit_qureg_extract_index.jeff"},
        RoundTripTestCase{"unit_qureg_extract_slice.jeff"},
        RoundTripTestCase{"unit_qureg_free_zero.jeff"},
        RoundTripTestCase{"unit_qureg_free.jeff"},
        RoundTripTestCase{"unit_qureg_insert_index.jeff"},
        RoundTripTestCase{"unit_qureg_insert_slice.jeff"},
        RoundTripTestCase{"unit_qureg_join.jeff"},
        RoundTripTestCase{"unit_qureg_split.jeff"},
        // Int operations
        RoundTripTestCase{"unit_int_abs.jeff"},
        RoundTripTestCase{"unit_int_add.jeff"},
        RoundTripTestCase{"unit_int_and.jeff"},
        RoundTripTestCase{"unit_int_const1.jeff"},
        RoundTripTestCase{"unit_int_const8.jeff"},
        RoundTripTestCase{"unit_int_const16.jeff"},
        RoundTripTestCase{"unit_int_const32.jeff"},
        RoundTripTestCase{"unit_int_const64.jeff"},
        RoundTripTestCase{"unit_int_divS.jeff"},
        RoundTripTestCase{"unit_int_divU.jeff"},
        RoundTripTestCase{"unit_int_eq.jeff"},
        RoundTripTestCase{"unit_int_lteS.jeff"},
        RoundTripTestCase{"unit_int_lteU.jeff"},
        RoundTripTestCase{"unit_int_ltS.jeff"},
        RoundTripTestCase{"unit_int_ltU.jeff"},
        RoundTripTestCase{"unit_int_maxS.jeff"},
        RoundTripTestCase{"unit_int_maxU.jeff"},
        RoundTripTestCase{"unit_int_minS.jeff"},
        RoundTripTestCase{"unit_int_minU.jeff"},
        RoundTripTestCase{"unit_int_mul.jeff"},
        RoundTripTestCase{"unit_int_not.jeff"},
        RoundTripTestCase{"unit_int_or.jeff"},
        RoundTripTestCase{"unit_int_pow.jeff"},
        RoundTripTestCase{"unit_int_remS.jeff"},
        RoundTripTestCase{"unit_int_remU.jeff"},
        RoundTripTestCase{"unit_int_shl.jeff"},
        RoundTripTestCase{"unit_int_shr.jeff"},
        RoundTripTestCase{"unit_int_xor.jeff"},
        // IntArray operations
        RoundTripTestCase{"unit_int_array_const1.jeff"},
        RoundTripTestCase{"unit_int_array_const8.jeff"},
        RoundTripTestCase{"unit_int_array_const16.jeff"},
        RoundTripTestCase{"unit_int_array_const32.jeff"},
        RoundTripTestCase{"unit_int_array_const64.jeff"},
        RoundTripTestCase{"unit_int_array_create.jeff"},
        RoundTripTestCase{"unit_int_array_get_index.jeff"},
        RoundTripTestCase{"unit_int_array_length.jeff"},
        RoundTripTestCase{"unit_int_array_set_index.jeff"},
        RoundTripTestCase{"unit_int_array_zero.jeff"},
        // Float operations
        RoundTripTestCase{"unit_float_abs.jeff"},
        RoundTripTestCase{"unit_float_acos.jeff"},
        RoundTripTestCase{"unit_float_acosh.jeff"},
        RoundTripTestCase{"unit_float_add.jeff"},
        RoundTripTestCase{"unit_float_asin.jeff"},
        RoundTripTestCase{"unit_float_asinh.jeff"},
        RoundTripTestCase{"unit_float_atan.jeff"},
        RoundTripTestCase{"unit_float_atan2.jeff"},
        RoundTripTestCase{"unit_float_atanh.jeff"},
        RoundTripTestCase{"unit_float_ceil.jeff"},
        RoundTripTestCase{"unit_float_const32.jeff"},
        RoundTripTestCase{"unit_float_const64.jeff"},
        RoundTripTestCase{"unit_float_cos.jeff"},
        RoundTripTestCase{"unit_float_cosh.jeff"},
        RoundTripTestCase{"unit_float_eq.jeff"},
        RoundTripTestCase{"unit_float_exp.jeff"},
        RoundTripTestCase{"unit_float_floor.jeff"},
        RoundTripTestCase{"unit_float_isInf.jeff"},
        RoundTripTestCase{"unit_float_isNan.jeff"},
        RoundTripTestCase{"unit_float_log.jeff"},
        RoundTripTestCase{"unit_float_lt.jeff"},
        RoundTripTestCase{"unit_float_lte.jeff"},
        RoundTripTestCase{"unit_float_max.jeff"},
        RoundTripTestCase{"unit_float_min.jeff"},
        RoundTripTestCase{"unit_float_mul.jeff"},
        RoundTripTestCase{"unit_float_pow.jeff"},
        RoundTripTestCase{"unit_float_sin.jeff"},
        RoundTripTestCase{"unit_float_sinh.jeff"},
        RoundTripTestCase{"unit_float_sqrt.jeff"},
        RoundTripTestCase{"unit_float_sub.jeff"},
        RoundTripTestCase{"unit_float_tan.jeff"},
        RoundTripTestCase{"unit_float_tanh.jeff"},
        // FloatArray operations
        RoundTripTestCase{"unit_float_array_const32.jeff"},
        RoundTripTestCase{"unit_float_array_const64.jeff"},
        RoundTripTestCase{"unit_float_array_create.jeff"},
        RoundTripTestCase{"unit_float_array_get_index.jeff"},
        RoundTripTestCase{"unit_float_array_length.jeff"},
        RoundTripTestCase{"unit_float_array_set_index.jeff"},
        RoundTripTestCase{"unit_float_array_zero.jeff"},
        // SCF operations
        RoundTripTestCase{"unit_scf_do_while.jeff"},
        RoundTripTestCase{"unit_scf_for.jeff"},
        RoundTripTestCase{"unit_scf_switch.jeff"},
        RoundTripTestCase{"unit_scf_while.jeff"},
        // Integration tests
        RoundTripTestCase{"bell_pair.jeff"},
        // RoundTripTestCase{"catalyst_simple.jeff"},
        // RoundTripTestCase{"catalyst_tket_opt.jeff"},
        // RoundTripTestCase{"entangled_calls.jeff"},
        // RoundTripTestCase{"entangled_qs.jeff"},
        RoundTripTestCase{"python_optimization.jeff"}));
