// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef JEFF_INTERFACES
#define JEFF_INTERFACES

include "mlir/IR/OpBase.td"

def QubitOperation : OpInterface<"QubitOperation"> {
    let description = [{
    }];

    let cppNamespace = "::jeff";

    let methods = [];

    let verify = [{}];
}

def QuregOperation : OpInterface<"QuregOperation"> {
    let description = [{
    }];

    let cppNamespace = "::jeff";

    let methods = [];

    let verify = [{}];
}

def QubitGate : OpInterface<"QubitGate"> {
    let description = [{
    }];

    let cppNamespace = "::jeff";

    let methods = [];

    let verify = [{}];
}

def MultipleCtrlQubitsGate : OpInterface<"MultipleCtrlQubitsGate", [QubitGate]> {
    let description = [{
    }];

    let cppNamespace = "::jeff";

    let methods = [];

    let verify = [{
        auto gate = mlir::cast<ConcreteOp>($_op);

        auto provided_in_ctrl_qubits = gate.getInCtrlQubits().size();
        auto expected_ctrl_qubits = gate.getNumCtrls();
        if (provided_in_ctrl_qubits != expected_ctrl_qubits) {
            return $_op->emitOpError(
                "The number of provided input control qubits does not match the expected amount");
        }

        auto provided_out_ctrl_qubits = gate.getOutCtrlQubits().size();
        if (provided_out_ctrl_qubits != expected_ctrl_qubits) {
            return $_op->emitOpError(
                "The number of provided output control qubits does not match the expected amount");
        }

        return mlir::success();
    }];
}

def MultiplePauliGate : OpInterface<"MultiplePauliGate", [QubitGate]> {
    let description = [{
    }];

    let cppNamespace = "::jeff";

    let methods = [];

    let verify = [{
        auto gate = mlir::cast<ConcreteOp>($_op);

        auto provided_in_qubits = gate.getInQubits().size();
        auto expected_qubits = static_cast<size_t>(gate.getPauliGatesAttr().size());
        if (provided_in_qubits != expected_qubits) {
            return $_op->emitOpError(
                "The number of provided input qubits does not match the expected amount");
        }

        auto provided_out_qubits = gate.getOutQubits().size();
        if (provided_out_qubits != expected_qubits) {
            return $_op->emitOpError(
                "The number of provided output qubits does not match the expected amount");
        }

        return mlir::success();
    }];
}

def MultipleQubitsGate : OpInterface<"MultipleQubitsGate", [QubitGate]> {
    let description = [{
    }];

    let cppNamespace = "::jeff";

    let methods = [];

    let verify = [{
        auto gate = mlir::cast<ConcreteOp>($_op);

        auto provided_in_qubits = gate.getInQubits().size();
        auto expected_qubits = gate.getNumQubits();
        if (provided_in_qubits != expected_qubits) {
            return $_op->emitOpError(
                "The number of provided input qubits does not match the expected amount");
        }

        auto provided_out_qubits = gate.getOutQubits().size();
        if (provided_out_qubits != expected_qubits) {
            return $_op->emitOpError(
                "The number of provided output qubits does not match the expected amount");
        }

        return mlir::success();
    }];
}

def MultipleParamsGate : OpInterface<"MultipleParamsGate", [QubitGate]> {
    let description = [{
    }];

    let cppNamespace = "::jeff";

    let methods = [];

    let verify = [{
        auto gate = mlir::cast<ConcreteOp>($_op);

        auto provided_params = gate.getParams().size();
        auto expected_params = gate.getNumParams();
        if (provided_params != expected_params) {
            return $_op->emitOpError(
                "The number of provided parameters does not match the expected amount");
        }

        return mlir::success();
    }];
}

def WellKnownGate : OpInterface<"WellKnownGate"> {
    let description = [{
    }];

    let cppNamespace = "::jeff";

    let methods = [];

    let verify = [{}];
}

def IntOperation : OpInterface<"IntOperation"> {
    let description = [{
    }];

    let cppNamespace = "::jeff";

    let methods = [];

    let verify = [{}];
}

def IntArrayOperation : OpInterface<"IntArrayOperation"> {
    let description = [{
    }];

    let cppNamespace = "::jeff";

    let methods = [];

    let verify = [{}];
}

def FloatOperation : OpInterface<"FloatOperation"> {
    let description = [{
    }];

    let cppNamespace = "::jeff";

    let methods = [];

    let verify = [{}];
}

def FloatArrayOperation : OpInterface<"FloatArrayOperation"> {
    let description = [{
    }];

    let cppNamespace = "::jeff";

    let methods = [];

    let verify = [{}];
}

def SCFOperation : OpInterface<"SCFOperation"> {
    let description = [{
    }];

    let cppNamespace = "::jeff";

    let methods = [];

    let verify = [{}];
}

#endif // JEFF_INTERFACES
