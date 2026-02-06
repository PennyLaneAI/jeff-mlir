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

#ifndef JEFF_OPS
#define JEFF_OPS

include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "jeff/IR/JeffDialect.td"
include "jeff/IR/JeffInterfaces.td"

//===----------------------------------------------------------------------===//
// Qubit operations.
//===----------------------------------------------------------------------===//

class Qubit_Op<string mnemonic, list<Trait> traits = []> : Jeff_Op<mnemonic, traits # [QubitOperation]>;

def QubitAllocOp : Qubit_Op<"qubit_alloc", []> {
    let summary = "Allocates a new qubit in the |0> state.";
    let description = [{}];

    let results = (outs
        QubitType:$out_qubit
    );

    // %0 = jeff.qubit_alloc : !jeff.qubit
    let assemblyFormat = [{
        attr-dict `:` type(results)
    }];
}

def QubitFreeOp : Qubit_Op<"qubit_free", []> {
    let summary = "Frees a qubit.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit
    );

    // jeff.qubit_free %0 : !jeff.qubit
    let assemblyFormat = [{
        $in_qubit attr-dict `:` type($in_qubit)
    }];
}

def QubitFreeZeroOp : Qubit_Op<"qubit_free_zero", []> {
    let summary = "Frees a qubit in the |0> state.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit
    );

    // jeff.qubit_free_zero %0 : !jeff.qubit
    let assemblyFormat = [{
        $in_qubit attr-dict `:` type($in_qubit)
    }];
}

def QubitMeasureOp : Qubit_Op<"qubit_measure", []> {
    let summary = "Perform a destructive measurement of a qubit in the computational basis.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit
    );

    let results = (outs
        I1:$result
    );

    // %1 = jeff.qubit_measure %0 : i1
    let assemblyFormat = [{
        $in_qubit attr-dict `:` type(results)
    }];
}

def QubitMeasureNDOp : Qubit_Op<"qubit_measure_nd", []> {
    let summary = "Perform a non-destructive measurement of a qubit in the computational basis.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit
    );

    let results = (outs
        I1:$result,
        QubitType:$out_qubit
    );

    // %1, %2 = jeff.qubit_measure_nd %0 : i1, !jeff.qubit
    let assemblyFormat = [{
        $in_qubit attr-dict `:` type(results)
    }];
}

def QubitResetOp : Qubit_Op<"qubit_reset", [SameOperandsAndResultType]> {
    let summary = "Resets a qubit to the |0> state.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit
    );

    let results = (outs
        QubitType:$out_qubit
    );

    // %1 = jeff.qubit_reset %0 : !jeff.qubit
    let assemblyFormat = [{
        $in_qubit attr-dict `:` type(results)
    }];
}

//===----------------------------------------------------------------------===//
// Gate operations.
//===----------------------------------------------------------------------===//

class Gate_Op<string mnemonic, list<Trait> traits = []> : Qubit_Op<mnemonic, traits # [MultipleCtrlQubitsGate]>;

def PPROp : Gate_Op<"ppr", [AttrSizedOperandSegments, AttrSizedResultSegments, MultiplePauliGate]> {
    let summary = "Apply an arbitrary Pauli-product rotation gate.";
    let description = [{}];

    let arguments = (ins
        Variadic<QubitType>:$in_qubits,
        Variadic<QubitType>:$in_ctrl_qubits,
        F32:$rotation,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power,
        DenseI32ArrayAttr:$pauli_gates
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::ValueRange":$in_qubits,
             "mlir::ValueRange":$in_ctrl_qubits,
             "mlir::Value":$rotation,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power,
             "mlir::DenseI32ArrayAttr":$pauli_gates
        ),[{
            PPROp::build($_builder, $_state,
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubits=*/ in_qubits,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*rotation=*/ rotation,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power,
            /*pauli_gates=*/ pauli_gates
            );
        }]>,
    ];

    // %4, %5 = jeff.ppr (%0, %1) %2 ctrls(%3) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        `(` $rotation `,` $pauli_gates `)` attr-dict $in_qubits ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubits) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def CustomOp : Gate_Op<"custom", [AttrSizedOperandSegments, AttrSizedResultSegments, MultipleQubitsGate, MultipleParamsGate]> {
    let summary = "Apply an arbitrary quantum gate.";
    let description = [{}];

    let arguments = (ins
        Variadic<QubitType>:$in_qubits,
        Variadic<QubitType>:$in_ctrl_qubits,
        Variadic<F32>:$params,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power,
        StrAttr:$name,
        I8Attr:$num_qubits,
        I8Attr:$num_params
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::ValueRange":$in_qubits,
             "mlir::ValueRange":$in_ctrl_qubits,
             "mlir::ValueRange":$params,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power,
             "mlir::StringRef":$name,
             "uint8_t":$num_qubits,
             "uint8_t":$num_params
        ),[{
            CustomOp::build($_builder, $_state,
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubits=*/ in_qubits,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*params=*/ params,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power,
            /*name=*/ name,
            /*num_qubits=*/ num_qubits,
            /*num_params=*/ num_params
            );
        }]>,
    ];

    // %3, %4 = jeff.custom "Rot"(%0) %1 ctrls(%2) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        $name `(` $params `)` attr-dict $in_qubits ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubits) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

//===----------------------------------------------------------------------===//
// Well Known Gate operations.
//===----------------------------------------------------------------------===//

class WellKnownGate_Op<string mnemonic, list<Trait> traits = []> : Gate_Op<mnemonic, traits # [WellKnownGate]>;

def GPhaseOp : WellKnownGate_Op<"gphase", []> {
    let summary = "Global phase operation on the 'vacuum' state (no qubits).";
    let description = [{}];

    let arguments = (ins
        F64:$rotation,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$rotation,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            GPhaseOp::build($_builder, $_state,
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*rotation=*/ rotation,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %2 = jeff.gphase (%0) ctrls(%1) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        `(` $rotation `)` attr-dict ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def IOp : WellKnownGate_Op<"i", []> {
    let summary = "Identity (no-op) gate. Pi radians X rotation.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            IOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %2, %3 = jeff.i %0 ctrls(%1) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def XOp : WellKnownGate_Op<"x", []> {
    let summary = "Pauli-X (NOT) gate. Pi radians X rotation.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            XOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %2, %3 = jeff.x %0 ctrls(%1) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def YOp : WellKnownGate_Op<"y", []> {
    let summary = "Pauli-Y gate. Pi radians Y rotation.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            YOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %2, %3 = jeff.y %0 ctrls(%1) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def ZOp : WellKnownGate_Op<"z", []> {
    let summary = "Pauli-Z gate. Pi radians Z rotation.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            ZOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %2, %3 = jeff.z %0 ctrls(%1) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def SOp : WellKnownGate_Op<"s", []> {
    let summary = "Root-Z gate. Pi/2 radians Z rotation.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            SOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %2, %3 = jeff.s %0 ctrls(%1) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def TOp : WellKnownGate_Op<"t", []> {
    let summary = "T gate. Pi/4 radians Z rotation.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            TOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %2, %3 = jeff.t %0 ctrls(%1) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def R1Op : WellKnownGate_Op<"r1", []> {
    let summary = "Rotation about the |1> state.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        F64:$rotation,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::Value":$rotation,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            R1Op::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*rotation=*/ rotation,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %3, %4 = jeff.r1 (%0) %1 ctrls(%2) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        `(` $rotation `)` attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def RxOp : WellKnownGate_Op<"rx", []> {
    let summary = "Rotation about the X axis.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        F64:$rotation,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::Value":$rotation,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            RxOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*rotation=*/ rotation,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %3, %4 = jeff.rx (%0) %1 ctrls(%2) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        `(` $rotation `)` attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def RyOp : WellKnownGate_Op<"ry", []> {
    let summary = "Rotation about the Y axis.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        F64:$rotation,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::Value":$rotation,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            RyOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*rotation=*/ rotation,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %3, %4 = jeff.ry (%0) %1 ctrls(%2) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        `(` $rotation `)` attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def RzOp : WellKnownGate_Op<"rz", []> {
    let summary = "Rotation about the Z axis.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        F64:$rotation,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::Value":$rotation,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            RzOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*rotation=*/ rotation,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %3, %4 = jeff.rz (%0) %1 ctrls(%2) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        `(` $rotation `)` attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def HOp : WellKnownGate_Op<"h", []> {
    let summary = "Hadamard gate.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            HOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %2, %3 = jeff.h %0 ctrls(%1) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def UOp : WellKnownGate_Op<"u", []> {
    let summary = "Euler gate.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit,
        F64:$theta,
        F64:$phi,
        F64:$lambda,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let results = (outs
        QubitType:$out_qubit,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit,
             "mlir::Value":$theta,
             "mlir::Value":$phi,
             "mlir::Value":$lambda,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            UOp::build($_builder, $_state,
            /*out_qubit=*/ in_qubit.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit=*/ in_qubit,
            /*theta=*/ theta,
            /*phi=*/ phi,
            /*lambda=*/ lambda,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    // %5, %6 = jeff.u (%0, %1, %2) %3 ctrls(%4) : !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        `(` $theta `,` $phi `,` $lambda `)` attr-dict $in_qubit ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

def SwapOp : WellKnownGate_Op<"swap", []> {
    let summary = "Swap gate. Swaps the state of two qubits.";
    let description = [{}];

    let arguments = (ins
        QubitType:$in_qubit_one,
        QubitType:$in_qubit_two,
        Variadic<QubitType>:$in_ctrl_qubits,
        I8Attr:$num_ctrls,
        I1Attr:$is_adjoint,
        I8Attr:$power
    );

    let builders = [
        OpBuilder<
        // Convenience builder for a gate with parameters and controls
        // Note that number of out_qubits = number of in_qubits,
        // and number of out_ctrl_qubits = number of in_ctrl_qubits
        (ins
             "mlir::Value":$in_qubit_one,
             "mlir::Value":$in_qubit_two,
             "mlir::ValueRange":$in_ctrl_qubits,
             "uint8_t":$num_ctrls,
             "bool":$is_adjoint,
             "uint8_t":$power
        ),[{
            SwapOp::build($_builder, $_state,
            /*out_qubit_one=*/ in_qubit_one.getType(),
            /*out_qubit_two=*/ in_qubit_two.getType(),
            /*out_ctrl_qubits=*/ mlir::TypeRange(in_ctrl_qubits),
            /*in_qubit_one=*/ in_qubit_one,
            /*in_qubit_two=*/ in_qubit_two,
            /*in_ctrl_qubits=*/ in_ctrl_qubits,
            /*num_ctrls=*/ num_ctrls,
            /*is_adjoint=*/ is_adjoint,
            /*power=*/ power
            );
        }]>,
    ];

    let results = (outs
        QubitType:$out_qubit_one,
        QubitType:$out_qubit_two,
        Variadic<QubitType>:$out_ctrl_qubits
    );

    // %3, %4, %5 = jeff.swap %0 %1 ctrls(%2) : !quantum.bit !quantum.bit ctrls !quantum.bit
    let assemblyFormat = [{
        attr-dict $in_qubit_one $in_qubit_two ( `ctrls` `(` $in_ctrl_qubits^ `)` )? `:` type($out_qubit_one) type($out_qubit_two) (`ctrls` type($out_ctrl_qubits)^ )?
    }];
}

//===----------------------------------------------------------------------===//
// Qureg operations.
//===----------------------------------------------------------------------===//

class Qureg_Op<string mnemonic, list<Trait> traits = []> : Jeff_Op<mnemonic, traits # [QuregOperation]>;

def QuregAllocOp : Qubit_Op<"qureg_alloc", []> {
    let summary = "Allocates a new qubit register given a number of qubits in the |0> state.";
    let description = [{}];

    let arguments = (ins
        I32Attr:$num_qubits
    );

    let results = (outs
        QuregType:$qreg
    );

    // %1 = jeff.qureg_alloc (%0) : !jeff.qureg
    let assemblyFormat = [{
        `(` $num_qubits `)` attr-dict `:` type(results)
    }];
}

def QuregFreeOp : Qubit_Op<"qureg_free", []> {
    let summary = "Frees a qubit register.";
    let description = [{}];

    let arguments = (ins
        QuregType:$qreg
    );

    // jeff.qureg_free %0 : !jeff.qureg
    let assemblyFormat = [{
        $qreg attr-dict `:` type($qreg)
    }];
}

def QuregFreeZeroOp : Qubit_Op<"qureg_free_zero", []> {
    let summary = "Frees a qubit register, assuming that all qubits are in the |0> state.";
    let description = [{}];

    let arguments = (ins
        QuregType:$qreg
    );

    // jeff.qureg_free_zero %0 : !jeff.qureg
    let assemblyFormat = [{
        $qreg attr-dict `:` type($qreg)
    }];
}

def QuregExtractIndexOp : Qubit_Op<"qureg_extract_index", []> {
    let summary = "Extracts a single qubit from a qubit register.";
    let description = [{}];

    let arguments = (ins
        QuregType:$in_qreg,
        I32Attr:$index
    );

    let results = (outs
        QuregType:$out_qreg,
        QubitType:$out_qubit
    );

    // %2, %3 = jeff.qureg_extract_index (%0) %1 : !jeff.qureg !jeff.qubit
    let assemblyFormat = [{
        `(` $index `)` $in_qreg attr-dict `:` type(results)
    }];
}

def QuregInsertIndexOp : Qubit_Op<"qureg_insert_index", []> {
    let summary = "Insert a single qubit from a qubit register.";
    let description = [{}];

    let arguments = (ins
        QuregType:$in_qreg,
        QubitType:$in_qubit,
        I32Attr:$index
    );

    let results = (outs
        QuregType:$out_qreg
    );

    // %3 = jeff.qureg_insert_index (%0) %1 %2 : !jeff.qureg !jeff.qubit -> !jeff.qureg
    let assemblyFormat = [{
        `(` $index `)` $in_qreg $in_qubit attr-dict `:` functional-type(operands, results)
    }];
}

def QuregExtractSliceOp : Qubit_Op<"qureg_extract_slice", []> {
    let summary = "Extract a slice of qubits from a qubit register given a range of indices.";
    let description = [{}];

    let arguments = (ins
        QuregType:$in_qreg,
        I32Attr:$start,
        I32Attr:$length
    );

    let results = (outs
        QuregType:$out_qreg,
        QuregType:$new_qreg
    );

    // %3, %4 = jeff.qureg_extract_slice (%0, %1) %2 : !jeff.qureg !jeff.qureg
    let assemblyFormat = [{
        `(` $start `,` $length `)` $in_qreg attr-dict `:` type(results)
    }];
}

def QuregInsertSliceOp : Qubit_Op<"qureg_insert_slice", []> {
    let summary = "Insert a slice of qubits into a qubit register.";
    let description = [{}];

    let arguments = (ins
        QuregType:$in_qreg,
        QuregType:$new_qreg,
        I32Attr:$start
    );

    let results = (outs
        QuregType:$out_qreg
    );

    // %3 = jeff.qureg_insert_slice (%0) %1 %2 : !jeff.qureg !jeff.qureg -> !jeff.qureg
    let assemblyFormat = [{
        `(` $start `)` $in_qreg $new_qreg attr-dict `:` functional-type(operands, results)
    }];
}

def QuregLengthOp : Qubit_Op<"qureg_length", []> {
    let summary = "Returns the length of the qubit register.";
    let description = [{}];

    let arguments = (ins
        QuregType:$in_qreg
    );

    let results = (outs
        QuregType:$out_qreg,
        I32:$length
    );

    // %1, %2 = jeff.qureg_length %0 : !jeff.qureg i32
    let assemblyFormat = [{
        $in_qreg attr-dict `:` type(results)
    }];
}

def QuregSplitOp : Qubit_Op<"qureg_split", []> {
    let summary = "Splits a qubit register into two qubit registers at a given index.";
    let description = [{}];

    let arguments = (ins
        QuregType:$in_qreg,
        I32Attr:$index
    );

    let results = (outs
        QuregType:$out_qreg_one,
        QuregType:$out_qreg_two
    );

    // %1, %2 = jeff.qureg_split (%0) %1 : !jeff.qureg !jeff.qureg
    let assemblyFormat = [{
        `(` $index `)` $in_qreg attr-dict `:` type(results)
    }];
}

def QuregJoinOp : Qubit_Op<"qureg_join", [SameOperandsAndResultType]> {
    let summary = "Joins together two qubit registers into a single qubit register.";
    let description = [{}];

    let arguments = (ins
        QuregType:$in_qreg_one,
        QuregType:$in_qreg_two
    );

    let results = (outs
        QuregType:$out_qreg
    );

    // %2 = jeff.qureg_join %0 %1 : !jeff.qureg !jeff.qureg -> !jeff.qureg
    let assemblyFormat = [{
        $in_qreg_one $in_qreg_two attr-dict `:` functional-type(operands, results)
    }];
}

def QuregCreateOp : Qubit_Op<"qureg_create", []> {
    let summary = "Creates a qubit register from a variable number of input qubits.";
    let description = [{}];

    let arguments = (ins
        Variadic<QubitType>:$in_qubits
    );

    let results = (outs
        QuregType:$out_qreg
    );

    // %1 = jeff.qureg_create %0 : !jeff.qureg
    let assemblyFormat = [{
        $in_qubits attr-dict `:` type(results)
    }];
}

//===----------------------------------------------------------------------===//
// jeff dialect enums for numeric ops.
//===----------------------------------------------------------------------===//

def IntUnaryOperation : I32EnumAttr<"IntUnaryOperation",
    "Integer unary operations",
    [
        I32EnumAttrCase<"_not",   0>,
        I32EnumAttrCase<"_abs",   1>,
    ]> {
    let cppNamespace = "::mlir::jeff";
    let genSpecializedAttr = 0;
}

def IntBinaryOperation : I32EnumAttr<"IntBinaryOperation",
    "Integer binary operations",
    [
        I32EnumAttrCase<"_add",   0>,
        I32EnumAttrCase<"_sub",   1>,
        I32EnumAttrCase<"_mul",   2>,
        I32EnumAttrCase<"_divS",  3>,
        I32EnumAttrCase<"_divU",  4>,
        I32EnumAttrCase<"_pow",   5>,
        I32EnumAttrCase<"_and",   6>,
        I32EnumAttrCase<"_or",    7>,
        I32EnumAttrCase<"_xor",   8>,
        I32EnumAttrCase<"_minS",  9>,
        I32EnumAttrCase<"_minU",  10>,
        I32EnumAttrCase<"_maxS",  11>,
        I32EnumAttrCase<"_maxU",  12>,
        I32EnumAttrCase<"_remS",  13>,
        I32EnumAttrCase<"_remU",  14>,
        I32EnumAttrCase<"_shl",   15>,
        I32EnumAttrCase<"_shr",   16>,
    ]> {
    let cppNamespace = "::mlir::jeff";
    let genSpecializedAttr = 0;
}

def IntComparisonOperation : I32EnumAttr<"IntComparisonOperation",
    "Integer comparison operations",
    [
        I32EnumAttrCase<"_eq",   0>,
        I32EnumAttrCase<"_ltS",   1>,
        I32EnumAttrCase<"_lteS",   2>,
        I32EnumAttrCase<"_ltU",  3>,
        I32EnumAttrCase<"_lteU",  4>,
    ]> {
    let cppNamespace = "::mlir::jeff";
    let genSpecializedAttr = 0;
}

def FloatUnaryOperation : I32EnumAttr<"FloatUnaryOperation",
    "Float unary operations",
    [
        I32EnumAttrCase<"_sqrt",  0>,
        I32EnumAttrCase<"_abs",   1>,
        I32EnumAttrCase<"_ceil",  2>,
        I32EnumAttrCase<"_floor", 3>,
        I32EnumAttrCase<"_exp",   4>,
        I32EnumAttrCase<"_log",   5>,
        I32EnumAttrCase<"_sin",   6>,
        I32EnumAttrCase<"_cos",   7>,
        I32EnumAttrCase<"_tan",   8>,
        I32EnumAttrCase<"_asin",  9>,
        I32EnumAttrCase<"_acos",  10>,
        I32EnumAttrCase<"_atan",  11>,
        I32EnumAttrCase<"_sinh",  12>,
        I32EnumAttrCase<"_cosh",  13>,
        I32EnumAttrCase<"_tanh",  14>,
        I32EnumAttrCase<"_asinh", 15>,
        I32EnumAttrCase<"_acosh", 16>,
        I32EnumAttrCase<"_atanh", 17>,
    ]> {
    let cppNamespace = "::mlir::jeff";
    let genSpecializedAttr = 0;
}

def FloatBinaryOperation : I32EnumAttr<"FloatBinaryOperation",
    "Float binary operations",
    [
        I32EnumAttrCase<"_add",   0>,
        I32EnumAttrCase<"_sub",   1>,
        I32EnumAttrCase<"_mul",   2>,
        I32EnumAttrCase<"_pow",   3>,
        I32EnumAttrCase<"_atan2", 4>,
        I32EnumAttrCase<"_max",   5>,
        I32EnumAttrCase<"_min",   6>,
    ]> {
    let cppNamespace = "::mlir::jeff";
    let genSpecializedAttr = 0;
}

def FloatComparisonOperation : I32EnumAttr<"FloatComparisonOperation",
    "Float comparison operations",
    [
        I32EnumAttrCase<"_eq",   0>,
        I32EnumAttrCase<"_lt",   1>,
        I32EnumAttrCase<"_lte",   2>,
    ]> {
    let cppNamespace = "::mlir::jeff";
    let genSpecializedAttr = 0;
}

def FloatIsOperation : I32EnumAttr<"FloatIsOperation",
    "Float Is operations",
    [
        I32EnumAttrCase<"_isNan",  0>,
        I32EnumAttrCase<"_isInf",  1>,
    ]> {
    let cppNamespace = "::mlir::jeff";
    let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// jeff dialect attributes for numeric ops.
//===----------------------------------------------------------------------===//

def IntUnaryOperationAttr : EnumAttr<JeffDialect, IntUnaryOperation, "int_unary_operation">;

def IntBinaryOperationAttr : EnumAttr<JeffDialect, IntBinaryOperation, "int_binary_operation">;

def IntComparisonOperationAttr : EnumAttr<JeffDialect, IntComparisonOperation, "int_comparison_operation">;

def FloatUnaryOperationAttr : EnumAttr<JeffDialect, FloatUnaryOperation, "float_unary_operation">;

def FloatBinaryOperationAttr : EnumAttr<JeffDialect, FloatBinaryOperation, "float_binary_operation">;

def FloatComparisonOperationAttr : EnumAttr<JeffDialect, FloatComparisonOperation, "float_comparison_operation">;

def FloatIsOperationAttr : EnumAttr<JeffDialect, FloatIsOperation, "float_is_operation">;

//===----------------------------------------------------------------------===//
// Int operations.
//===----------------------------------------------------------------------===//

class Int_Op<string mnemonic, list<Trait> traits = []> : Jeff_Op<mnemonic, traits # [IntOperation]>;

def IntConst1Op : Int_Op<"int_const1", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I1Attr:$val
    );

    let results = (outs
        I1:$constant
    );

    // %1 = jeff.int_const1 (%0) : i1
    let assemblyFormat = [{
        `(` $val `)` attr-dict `:` type(results)
    }];
}

def IntConst8Op : Int_Op<"int_const8", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I8Attr:$val
    );

    let results = (outs
        I8:$constant
    );

    // %1 = jeff.int_const8 (%0) : i8
    let assemblyFormat = [{
        `(` $val `)` attr-dict `:` type(results)
    }];
}

def IntConst16Op : Int_Op<"int_const16", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I16Attr:$val
    );

    let results = (outs
        I16:$constant
    );

    // %1 = jeff.int_const16 (%0) : i16
    let assemblyFormat = [{
        `(` $val `)` attr-dict `:` type(results)
    }];
}

def IntConst32Op : Int_Op<"int_const32", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I32Attr:$val
    );

    let results = (outs
        I32:$constant
    );

    // %1 = jeff.int_const32 (%0) : i32
    let assemblyFormat = [{
        `(` $val `)` attr-dict `:` type(results)
    }];
}

def IntConst64Op : Int_Op<"int_const64", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I64Attr:$val
    );

    let results = (outs
        I64:$constant
    );

    // %1 = jeff.int_const64 (%0) : i64
    let assemblyFormat = [{
        `(` $val `)` attr-dict `:` type(results)
    }];
}

def IntUnaryOp : Int_Op<"int_unary_op", [SameOperandsAndResultType]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedIntType:$a,
        IntUnaryOperationAttr:$op
    );

    let results = (outs
        SupportedIntType:$b
    );

    // %1 = jeff.int_unary_op [0] (%0) : i64
    let assemblyFormat = [{
        `[` $op `]` `(` $a `)` attr-dict `:` type($a) type(results)
    }];
}

def IntBinaryOp : Int_Op<"int_binary_op", [SameOperandsAndResultType]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedIntType:$a,
        SupportedIntType:$b,
        IntBinaryOperationAttr:$op
    );

    let results = (outs
        SupportedIntType:$c
    );

    // %2 = jeff.int_binary_op [0] (%0, %1) : i64
    let assemblyFormat = [{
        `[` $op `]` `(` $a `,` $b `)` attr-dict `:` type($a) type($b) type(results)
    }];
}

def IntComparisonOp : Int_Op<"int_comp_op", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedIntType:$a,
        SupportedIntType:$b,
        IntComparisonOperationAttr:$op
    );

    let results = (outs
        I1:$c
    );

    // %2 = jeff.int_comp_op [0] (%0, %1) : i1
    let assemblyFormat = [{
        `[` $op `]` `(` $a `,` $b `)` attr-dict `:` type($a) type($b)
    }];
}

//===----------------------------------------------------------------------===//
// IntArray operations.
//===----------------------------------------------------------------------===//

class IntArray_Op<string mnemonic, list<Trait> traits = []> : Jeff_Op<mnemonic, traits # [IntArrayOperation]>;

def IntArrayConst1Op : IntArray_Op<"int_array_const1", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        DenseBoolArrayAttr:$in_array
    );

    let results = (outs
        1DTensorOf<[I1]>:$out_array
    );

    // %1 = jeff.int_array_const1 (%0) : i1
    let assemblyFormat = [{
        `(` $in_array `)` attr-dict `:` type(results)
    }];
}

def IntArrayConst8Op : IntArray_Op<"int_array_const8", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        DenseI8ArrayAttr:$in_array
    );

    let results = (outs
        1DTensorOf<[I8]>:$out_array
    );

    // %1 = jeff.int_array_const8 (%0) : i8
    let assemblyFormat = [{
        `(` $in_array `)` attr-dict `:` type(results)
    }];
}

def IntArrayConst16Op : IntArray_Op<"int_array_const16", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        DenseI16ArrayAttr:$in_array
    );

    let results = (outs
        1DTensorOf<[I16]>:$out_array
    );

    // %1 = jeff.int_array_const16 (%0) : i16
    let assemblyFormat = [{
        `(` $in_array `)` attr-dict `:` type(results)
    }];
}

def IntArrayConst32Op : IntArray_Op<"int_array_const32", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        DenseI32ArrayAttr:$in_array
    );

    let results = (outs
        1DTensorOf<[I32]>:$out_array
    );

    // %1 = jeff.int_array_const32 (%0) : i32
    let assemblyFormat = [{
        `(` $in_array `)` attr-dict `:` type(results)
    }];
}

def IntArrayConst64Op : IntArray_Op<"int_array_const64", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        DenseI64ArrayAttr:$in_array
    );

    let results = (outs
        1DTensorOf<[I64]>:$out_array
    );

    // %1 = jeff.int_array_const64 (%0) : i64
    let assemblyFormat = [{
        `(` $in_array `)` attr-dict `:` type(results)
    }];
}

def IntArrayZeroOp : IntArray_Op<"int_array_zero", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I32Attr:$length
    );

    let results = (outs
        1DTensorOf<[I1, I8, I16, I32, I64]>:$out_array
    );

    // %1 = jeff.int_array_zero (%0) : i64
    let assemblyFormat = [{
        `(` $length `)` attr-dict `:` type(results)
    }];
}

def IntArrayGetIndexOp : IntArray_Op<"int_array_get_index", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        1DTensorOf<[I1, I8, I16, I32, I64]>:$in_array,
        I32:$index
    );

    let results = (outs
        SupportedIntType:$value
    );

    // %2 = jeff.int_array_get_index (%1) %0 : i32 tensor<5xi64> -> tensor<5xi64>
    let assemblyFormat = [{
        `(` $index `)` $in_array attr-dict `:` type($index) type($in_array) `->` type(results)
    }];
}

def IntArraySetIndexOp : IntArray_Op<"int_array_set_index", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        1DTensorOf<[I1, I8, I16, I32, I64]>:$in_array,
        I32:$index,
        SupportedIntType:$value
    );

    let results = (outs
        1DTensorOf<[I1, I8, I16, I32, I64]>:$out_array
    );

    // %3 = jeff.int_array_set_index (%1) %0 %2 : i32 tensor<5xi64> i64 -> tensor<5xi64>
    let assemblyFormat = [{
        `(` $index `)` $in_array $value attr-dict `:` type($index) type($in_array) type($value) `->` type($out_array)
    }];
}

def IntArrayLengthOp : IntArray_Op<"int_array_length", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        1DTensorOf<[I1, I8, I16, I32, I64]>:$in_array
    );

    let results = (outs
        I32:$length
    );

    // %1 = jeff.int_array_length %0 : i64
    let assemblyFormat = [{
        $in_array attr-dict `:` type($in_array)
    }];
}

def IntArrayCreateOp : IntArray_Op<"int_array_create", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedIntArrayAttrType:$in_array
    );

    let results = (outs
        1DTensorOf<[I1, I8, I16, I32, I64]>:$out_array
    );

    // %1 = jeff.int_array_create %0 : i64
    let assemblyFormat = [{
        attr-dict `:` type(results)
    }];
}

//===----------------------------------------------------------------------===//
// Float operations.
//===----------------------------------------------------------------------===//

class Float_Op<string mnemonic, list<Trait> traits = []> : Jeff_Op<mnemonic, traits # [FloatOperation]>;

def FloatConst32Op : Float_Op<"float_const32", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        F32Attr:$val
    );

    let results = (outs
        F32:$constant
    );

    // %1 = jeff.float_const32 (%0) : f32
    let assemblyFormat = [{
        `(` $val `)` attr-dict `:` type(results)
    }];
}

def FloatConst64Op : Float_Op<"float_const64", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        F64Attr:$val
    );

    let results = (outs
        F64:$constant
    );

    // %1 = jeff.float_const64 (%0) : f64
    let assemblyFormat = [{
        `(` $val `)` attr-dict `:` type(results)
    }];
}

def FloatUnaryOp : Float_Op<"float_unary_op", [SameOperandsAndResultType]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedFloatType:$a,
        FloatUnaryOperationAttr:$op
    );

    let results = (outs
        SupportedFloatType:$b
    );

    // %1 = jeff.float_unary_op [0] (%0) : f64
    let assemblyFormat = [{
       `[` $op `]` `(` $a `)` attr-dict `:` type($a) type(results)
    }];
}

def FloatBinaryOp : Float_Op<"float_binary_op", [SameOperandsAndResultType]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedFloatType:$a,
        SupportedFloatType:$b,
        FloatBinaryOperationAttr:$op
    );

    let results = (outs
        SupportedFloatType:$c
    );

    // %2 = jeff.float_binary_op [0] (%0, %1) : f64
    let assemblyFormat = [{
       `[` $op `]` `(` $a `,` $b `)` attr-dict `:` type($a) type($b) type(results)
    }];
}

def FloatComparisonOp : Float_Op<"float_comp_op", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedFloatType:$a,
        SupportedFloatType:$b,
        FloatComparisonOperationAttr:$op
    );

    let results = (outs
        I1:$c
    );

    // %2 = jeff.float_comp_op [0] (%0, %1) : i1
    let assemblyFormat = [{
       `[` $op `]` `(` $a `,` $b `)` attr-dict `:` type($a) type($b)
    }];
}

def FloatIsOp : Float_Op<"float_is_op", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedFloatType:$a,
        FloatIsOperationAttr:$op
    );

    let results = (outs
        I1:$b
    );

    // %1 = jeff.float_is_op [0] (%0) : i1
    let assemblyFormat = [{
       `[` $op `]` `(` $a `)` attr-dict `:` type($a)
    }];
}

//===----------------------------------------------------------------------===//
// FloatArray operations.
//===----------------------------------------------------------------------===//

class FloatArray_Op<string mnemonic, list<Trait> traits = []> : Jeff_Op<mnemonic, traits # [FloatArrayOperation]>;

def FloatArrayConst32Op : FloatArray_Op<"float_array_const32", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        DenseF32ArrayAttr:$in_array
    );

    let results = (outs
        1DTensorOf<[F32]>:$out_array
    );

    // %1 = jeff.float_array_const32 (%0) : f32
    let assemblyFormat = [{
        `(` $in_array `)` attr-dict `:` type(results)
    }];
}

def FloatArrayConst64Op : FloatArray_Op<"float_array_const64", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        DenseF64ArrayAttr:$in_array
    );

    let results = (outs
        1DTensorOf<[F64]>:$out_array
    );

    // %1 = jeff.float_array_const64 (%0) : f64
    let assemblyFormat = [{
        `(` $in_array `)` attr-dict `:` type(results)
    }];
}

def FloatArrayZeroOp : FloatArray_Op<"float_array_zero", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        I32Attr:$length
    );

    let results = (outs
        1DTensorOf<[F32, F64]>:$out_array
    );

    // %1 = jeff.float_array_zero (%0) : f64
    let assemblyFormat = [{
        `(` $length `)` attr-dict `:` type(results)
    }];
}

def FloatArrayGetIndexOp : FloatArray_Op<"float_array_get_index", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        1DTensorOf<[F32, F64]>:$in_array,
        I32Attr:$index
    );

    let results = (outs
        SupportedFloatType:$value
    );

    // %2 = jeff.float_array_get_index (%0) %1 : f64
    let assemblyFormat = [{
        `(` $index `)` $in_array attr-dict `:` type($in_array) type(results)
    }];
}

def FloatArraySetIndexOp : FloatArray_Op<"float_array_set_index", []> {
    let summary = "Set the value of an integer array at a given index.";
    let description = [{}];

    let arguments = (ins
        1DTensorOf<[F32, F64]>:$in_array,
        SupportedFloatType:$value,
        I32Attr:$index
    );

    let results = (outs
        1DTensorOf<[F32, F64]>:$out_array
    );

    // %3 = jeff.float_array_set_index (%0) %1 %2 : f64
    let assemblyFormat = [{
        `(` $index `)` $in_array $value attr-dict `:` type($in_array) type($value) type(results)
    }];
}

def FloatArrayLengthOp : FloatArray_Op<"float_array_length", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        1DTensorOf<[F32, F64]>:$in_array
    );

    let results = (outs
        I32:$length
    );

    // %1 = jeff.float_array_length %0 : i64
    let assemblyFormat = [{
        $in_array attr-dict `:` type($in_array)
    }];
}

def FloatArrayCreateOp : FloatArray_Op<"float_array_create", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedFloatArrayAttrType:$in_array
    );

    let results = (outs
        1DTensorOf<[F32, F64]>:$out_array
    );

    // %1 = jeff.float_array_create %0 : f64
    let assemblyFormat = [{
        attr-dict `:` type(results)
    }];
}

//===----------------------------------------------------------------------===//
// Structured control flow operations.
//===----------------------------------------------------------------------===//

class SCF_Op<string mnemonic, list<Trait> traits = []> : Jeff_Op<mnemonic, traits # [SCFOperation]>;

def SwitchOp : SCF_Op<"switch", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedIntType:$selection,
        Variadic<AnyType>:$in_values
    );

    let regions = (region
        MaxSizedRegion<1>:$default,
        VariadicRegion<SizedRegion<1>>:$branches
    );

    let results = (outs
        Variadic<AnyType>:$out_values
    );

    let assemblyFormat = [{
        `(` $selection `)` $branches ($default^)? $in_values attr-dict `:` type($selection) type($in_values) type($out_values)
    }];
}

def ForOp : SCF_Op<"for", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SupportedIntType:$start,
        SupportedIntType:$stop,
        SupportedIntType:$step,
        Variadic<AnyType>:$in_values
    );

    let regions = (region SizedRegion<1>:$body);

    let results = (outs
        Variadic<AnyType>:$out_values
    );

    let assemblyFormat = [{
        `(` $start `,` $stop `,` $step `)` $in_values attr-dict `:` type($start) type($stop) type($step) type($in_values) type($out_values) regions
    }];
}

def WhileOp : SCF_Op<"while", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        Variadic<AnyType>:$in_values
    );

    let regions = (region
        SizedRegion<1>:$condition,
        SizedRegion<1>:$body
    );

    let results = (outs
        Variadic<AnyType>:$out_values
    );

    let assemblyFormat = [{
        $in_values attr-dict `:` type($in_values) type($out_values) regions
    }];
}

def DoWhileOp : SCF_Op<"doWhile", []> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        Variadic<AnyType>:$in_values
    );

    let regions = (region
        SizedRegion<1>:$body,
        SizedRegion<1>:$condition
    );

    let results = (outs
        Variadic<AnyType>:$out_values
    );

    let assemblyFormat = [{
       $in_values attr-dict `:` type($in_values) type($out_values) regions
    }];
}

def YieldOp : SCF_Op<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["SwitchOp", "ForOp", "WhileOp", "DoWhileOp"]>]> {
  let summary = "Return results from SCF regions";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$out_values);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($out_values^ `:` type($out_values))? }];
}

#endif // JEFF_OPS
