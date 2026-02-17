from pathlib import Path
from typing import Callable
from jeff import (
    JeffRegion,
    JeffModule,
    IntType,
    JeffOp,
    JeffValue,
    QubitType,
    FunctionDef,
    qubit_alloc,
    qubit_free,
    FloatType,
    quantum_gate,
    pauli_rotation,
    QuregType,
    WhileSCF,
    DoWhileSCF,
)

# Registry for generator functions
_generators = []


def register_generator(function: Callable[[], None]) -> Callable[[], None]:
    """Decorator for registering generator functions."""
    _generators.append(function)
    return function


def _create_and_write_module(operations: list[JeffOp], output_filename: str) -> None:
    body = JeffRegion(
        sources=[],
        targets=[],
        operations=operations,
    )
    function = FunctionDef(name="main", body=body)
    module = JeffModule([function])

    output_file = Path(__file__).parent / output_filename
    output_file.unlink(missing_ok=True)
    module.write_out(output_file)


@register_generator
def generate_qubit_alloc() -> None:
    alloc = qubit_alloc()
    _create_and_write_module([alloc], "unit_qubit_alloc.jeff")


@register_generator
def generate_qubit_free() -> None:
    alloc = qubit_alloc()
    free = qubit_free(alloc.outputs[0])
    _create_and_write_module([alloc, free], "unit_qubit_free.jeff")


@register_generator
def generate_qubit_free_zero() -> None:
    alloc = qubit_alloc()
    free_zero = JeffOp("qubit", "freeZero", [alloc.outputs[0]], [])
    _create_and_write_module([alloc, free_zero], "unit_qubit_free_zero.jeff")


@register_generator
def generate_qubit_measure() -> None:
    alloc = qubit_alloc()
    measure = JeffOp(
        "qubit",
        "measure",
        [alloc.outputs[0]],
        [JeffValue(IntType(1))],
    )
    _create_and_write_module([alloc, measure], "unit_qubit_measure.jeff")


@register_generator
def generate_qubit_measure_nd() -> None:
    alloc = qubit_alloc()
    measure = JeffOp(
        "qubit",
        "measureNd",
        [alloc.outputs[0]],
        [JeffValue(QubitType()), JeffValue(IntType(1))],
    )
    free = qubit_free(measure.outputs[0])
    _create_and_write_module([alloc, measure, free], "unit_qubit_measure_nd.jeff")


@register_generator
def generate_qubit_reset() -> None:
    alloc = qubit_alloc()
    reset = JeffOp("qubit", "reset", [alloc.outputs[0]], [JeffValue(QubitType())])
    free = qubit_free(reset.outputs[0])
    _create_and_write_module([alloc, reset, free], "unit_qubit_reset.jeff")


@register_generator
def generate_one_qubit_zero_parameter() -> None:
    for gate_name in ["x", "y", "z", "h", "s", "t", "h", "i"]:
        alloc = qubit_alloc()
        gate = quantum_gate(gate_name, qubits=[alloc.outputs[0]])
        free = qubit_free(gate.outputs[0])
        _create_and_write_module([alloc, gate, free], f"unit_gate_{gate_name}.jeff")


@register_generator
def generate_multi_controlled_one_qubit_zero_parameter() -> None:
    for gate_name in ["x", "y", "z", "h", "s", "t", "h", "i"]:
        alloc1 = qubit_alloc()
        alloc2 = qubit_alloc()
        alloc3 = qubit_alloc()
        gate = quantum_gate(
            gate_name,
            qubits=[alloc1.outputs[0]],
            control_qubits=[alloc2.outputs[0], alloc3.outputs[0]],
        )
        free1 = qubit_free(gate.outputs[0])
        free2 = qubit_free(gate.outputs[1])
        free3 = qubit_free(gate.outputs[2])
        _create_and_write_module(
            [alloc1, alloc2, alloc3, gate, free1, free2, free3],
            f"unit_gate_mc{gate_name}.jeff",
        )


@register_generator
def generate_one_qubit_one_parameter() -> None:
    for gate_name in ["r1", "rx", "ry", "rz"]:
        alloc = qubit_alloc()
        rotation = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
        gate = quantum_gate(
            gate_name,
            qubits=[alloc.outputs[0]],
            params=[rotation.outputs[0]],
        )
        free = qubit_free(gate.outputs[0])
        _create_and_write_module(
            [alloc, rotation, gate, free],
            f"unit_gate_{gate_name}.jeff",
        )


@register_generator
def generate_controlled_one_qubit_one_parameter() -> None:
    for gate_name in ["r1", "rx", "ry", "rz"]:
        alloc1 = qubit_alloc()
        alloc2 = qubit_alloc()
        rotation = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
        gate = quantum_gate(
            gate_name,
            qubits=[alloc1.outputs[0]],
            params=[rotation.outputs[0]],
            control_qubits=[alloc2.outputs[0]],
        )
        free1 = qubit_free(gate.outputs[0])
        free2 = qubit_free(gate.outputs[1])
        _create_and_write_module(
            [alloc1, alloc2, rotation, gate, free1, free2],
            f"unit_gate_c{gate_name}.jeff",
        )


@register_generator
def generate_u() -> None:
    alloc = qubit_alloc()
    theta = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.1)
    phi = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.2)
    lambda_ = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.3)
    gate = quantum_gate(
        "u",
        qubits=[alloc.outputs[0]],
        params=[theta.outputs[0], phi.outputs[0], lambda_.outputs[0]],
    )
    free = qubit_free(gate.outputs[0])
    _create_and_write_module(
        [alloc, theta, phi, lambda_, gate, free], "unit_gate_u.jeff"
    )


@register_generator
def generate_cu() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    theta = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.1)
    phi = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.2)
    lambda_ = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.3)
    gate = quantum_gate(
        "u",
        qubits=[alloc1.outputs[0]],
        params=[theta.outputs[0], phi.outputs[0], lambda_.outputs[0]],
        control_qubits=[alloc2.outputs[0]],
    )
    free1 = qubit_free(gate.outputs[0])
    free2 = qubit_free(gate.outputs[1])
    _create_and_write_module(
        [alloc1, alloc2, theta, phi, lambda_, gate, free1, free2],
        "unit_gate_cu.jeff",
    )


@register_generator
def generate_swap() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    gate = quantum_gate("swap", qubits=[alloc1.outputs[0], alloc2.outputs[0]])
    free1 = qubit_free(gate.outputs[0])
    free2 = qubit_free(gate.outputs[1])
    _create_and_write_module(
        [alloc1, alloc2, gate, free1, free2], "unit_gate_swap.jeff"
    )


@register_generator
def generate_cswap() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    alloc3 = qubit_alloc()
    gate = quantum_gate(
        "swap",
        qubits=[alloc1.outputs[0], alloc2.outputs[0]],
        control_qubits=[alloc3.outputs[0]],
    )
    free1 = qubit_free(gate.outputs[0])
    free2 = qubit_free(gate.outputs[1])
    free3 = qubit_free(gate.outputs[2])
    _create_and_write_module(
        [alloc1, alloc2, alloc3, gate, free1, free2, free3],
        "unit_gate_cswap.jeff",
    )


@register_generator
def generate_gphase() -> None:
    alloc = qubit_alloc()
    rotation = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
    gate = quantum_gate(
        "gphase",
        qubits=[],
        params=[rotation.outputs[0]],
    )
    free = qubit_free(alloc.outputs[0])
    _create_and_write_module([alloc, rotation, gate, free], "unit_gate_gphase.jeff")


@register_generator
def generate_cgphase() -> None:
    alloc = qubit_alloc()
    rotation = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
    gate = quantum_gate(
        "gphase",
        qubits=[],
        params=[rotation.outputs[0]],
        control_qubits=[alloc.outputs[0]],
    )
    free = qubit_free(gate.outputs[0])
    _create_and_write_module([alloc, rotation, gate, free], "unit_gate_cgphase.jeff")


@register_generator
def generate_custom_1() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    gate = quantum_gate("custom", qubits=[alloc1.outputs[0], alloc2.outputs[0]])
    free1 = qubit_free(gate.outputs[0])
    free2 = qubit_free(gate.outputs[1])
    _create_and_write_module(
        [alloc1, alloc2, gate, free1, free2], "unit_gate_custom_1.jeff"
    )


@register_generator
def generate_custom_2() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    alloc3 = qubit_alloc()
    rotation = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
    gate = quantum_gate(
        "custom",
        qubits=[alloc1.outputs[0], alloc2.outputs[0]],
        params=[rotation.outputs[0]],
        control_qubits=[alloc3.outputs[0]],
    )
    free1 = qubit_free(gate.outputs[0])
    free2 = qubit_free(gate.outputs[1])
    free3 = qubit_free(gate.outputs[2])
    _create_and_write_module(
        [alloc1, alloc2, alloc3, rotation, gate, free1, free2, free3],
        "unit_gate_custom_2.jeff",
    )


@register_generator
def generate_ppr_rxx() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    rotation = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
    gate = pauli_rotation(
        angle=rotation.outputs[0],
        pauli_string=["x", "x"],
        qubits=[alloc1.outputs[0], alloc2.outputs[0]],
    )
    free1 = qubit_free(gate.outputs[0])
    free2 = qubit_free(gate.outputs[1])
    _create_and_write_module(
        [alloc1, alloc2, rotation, gate, free1, free2],
        "unit_gate_ppr_rxx.jeff",
    )


@register_generator
def generate_ppr_crxy() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    alloc3 = qubit_alloc()
    rotation = JeffOp("float", "const64", [], [JeffValue(FloatType(64))], 0.5)
    gate = pauli_rotation(
        angle=rotation.outputs[0],
        pauli_string=["x", "y"],
        qubits=[alloc1.outputs[0], alloc2.outputs[0]],
        control_qubits=[alloc3.outputs[0]],
    )
    free1 = qubit_free(gate.outputs[0])
    free2 = qubit_free(gate.outputs[1])
    free3 = qubit_free(gate.outputs[2])
    _create_and_write_module(
        [alloc1, alloc2, alloc3, rotation, gate, free1, free2, free3],
        "unit_gate_ppr_crxy.jeff",
    )


@register_generator
def generate_qureg_alloc() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType())],
    )
    _create_and_write_module([num_qubits, alloc], "unit_qureg_alloc.jeff")


@register_generator
def generate_qureg_free_zero() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType())],
    )
    free_zero = JeffOp("qureg", "freeZero", [alloc.outputs[0]], [])
    _create_and_write_module(
        [num_qubits, alloc, free_zero], "unit_qureg_free_zero.jeff"
    )


@register_generator
def generate_qureg_extract_index() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    index = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType())],
    )
    extract_index = JeffOp(
        "qureg",
        "extractIndex",
        [alloc.outputs[0], index.outputs[0]],
        [JeffValue(QuregType()), JeffValue(QubitType())],
    )
    free1 = JeffOp("qureg", "free", [extract_index.outputs[0]], [])
    free2 = qubit_free(extract_index.outputs[1])
    _create_and_write_module(
        [num_qubits, index, alloc, extract_index, free1, free2],
        "unit_qureg_extract_index.jeff",
    )


@register_generator
def generate_qureg_insert_index() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    index = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
    alloc1 = qubit_alloc()
    alloc2 = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType())],
    )
    insert_index = JeffOp(
        "qureg",
        "insertIndex",
        [alloc2.outputs[0], alloc1.outputs[0], index.outputs[0]],
        [JeffValue(QuregType())],
    )
    free = JeffOp("qureg", "free", [insert_index.outputs[0]], [])
    _create_and_write_module(
        [num_qubits, index, alloc1, alloc2, insert_index, free],
        "unit_qureg_insert_index.jeff",
    )


@register_generator
def generate_qureg_extract_slice() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    start = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 1)
    length = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 2)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType())],
    )
    extract_slice = JeffOp(
        "qureg",
        "extractSlice",
        [alloc.outputs[0], start.outputs[0], length.outputs[0]],
        [JeffValue(QuregType()), JeffValue(QuregType())],
    )
    free1 = JeffOp("qureg", "free", [extract_slice.outputs[0]], [])
    free2 = JeffOp("qureg", "free", [extract_slice.outputs[1]], [])
    _create_and_write_module(
        [num_qubits, start, length, alloc, extract_slice, free1, free2],
        "unit_qureg_extract_slice.jeff",
    )


@register_generator
def generate_qureg_insert_slice() -> None:
    num_qubits1 = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    num_qubits2 = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 2)
    index = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
    alloc1 = JeffOp(
        "qureg",
        "alloc",
        [num_qubits1.outputs[0]],
        [JeffValue(QuregType())],
    )
    alloc2 = JeffOp(
        "qureg",
        "alloc",
        [num_qubits2.outputs[0]],
        [JeffValue(QuregType())],
    )
    insert_slice = JeffOp(
        "qureg",
        "insertSlice",
        [alloc1.outputs[0], alloc2.outputs[0], index.outputs[0]],
        [JeffValue(QuregType())],
    )
    free = JeffOp("qureg", "free", [insert_slice.outputs[0]], [])
    _create_and_write_module(
        [num_qubits1, num_qubits2, index, alloc1, alloc2, insert_slice, free],
        "unit_qureg_insert_slice.jeff",
    )


@register_generator
def generate_qureg_length() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType())],
    )
    length = JeffOp(
        "qureg",
        "length",
        [alloc.outputs[0]],
        [JeffValue(QuregType()), JeffValue(IntType(32))],
    )
    free = JeffOp("qureg", "free", [length.outputs[0]], [])
    _create_and_write_module(
        [num_qubits, alloc, length, free],
        "unit_qureg_length.jeff",
    )


@register_generator
def generate_qureg_split() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    index = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType())],
    )
    split = JeffOp(
        "qureg",
        "split",
        [alloc.outputs[0], index.outputs[0]],
        [JeffValue(QuregType()), JeffValue(QuregType())],
    )
    free1 = JeffOp("qureg", "free", [split.outputs[0]], [])
    free2 = JeffOp("qureg", "free", [split.outputs[1]], [])
    _create_and_write_module(
        [num_qubits, index, alloc, split, free1, free2],
        "unit_qureg_split.jeff",
    )


@register_generator
def generate_qureg_join() -> None:
    num_qubits1 = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
    num_qubits2 = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 2)
    alloc1 = JeffOp(
        "qureg",
        "alloc",
        [num_qubits1.outputs[0]],
        [JeffValue(QuregType())],
    )
    alloc2 = JeffOp(
        "qureg",
        "alloc",
        [num_qubits2.outputs[0]],
        [JeffValue(QuregType())],
    )
    join = JeffOp(
        "qureg",
        "join",
        [alloc1.outputs[0], alloc2.outputs[0]],
        [JeffValue(QuregType())],
    )
    free = JeffOp("qureg", "free", [join.outputs[0]], [])
    _create_and_write_module(
        [num_qubits1, num_qubits2, alloc1, alloc2, join, free],
        "unit_qureg_join.jeff",
    )


@register_generator
def generate_qureg_create() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    alloc3 = qubit_alloc()
    qureg_create = JeffOp(
        "qureg",
        "create",
        [alloc1.outputs[0], alloc2.outputs[0], alloc3.outputs[0]],
        [JeffValue(QuregType())],
    )
    free = JeffOp("qureg", "free", [qureg_create.outputs[0]], [])
    _create_and_write_module(
        [alloc1, alloc2, alloc3, qureg_create, free],
        "unit_qureg_create.jeff",
    )


@register_generator
def generate_qureg_free() -> None:
    num_qubits = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 5)
    alloc = JeffOp(
        "qureg",
        "alloc",
        [num_qubits.outputs[0]],
        [JeffValue(QuregType())],
    )
    free = JeffOp("qureg", "free", [alloc.outputs[0]], [])
    _create_and_write_module([num_qubits, alloc, free], "unit_qureg_free.jeff")


@register_generator
def generate_scf_while() -> None:
    alloc = qubit_alloc()
    counter = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 0)

    three = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
    int_lt = JeffOp(
        "int",
        "ltS",
        [JeffValue(IntType(32)), three.outputs[0]],
        [JeffValue(IntType(1))],
    )
    condition = JeffRegion(
        sources=[JeffValue(QubitType()), int_lt.inputs[0]],
        targets=[int_lt.outputs[0]],
        operations=[three, int_lt],
    )

    h = quantum_gate("h", qubits=[JeffValue(QubitType())])
    one = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 1)
    int_add = JeffOp(
        "int", "add", [JeffValue(IntType(32)), one.outputs[0]], [JeffValue(IntType(32))]
    )
    body = JeffRegion(
        sources=[h.inputs[0], int_add.inputs[0]],
        targets=[h.outputs[0], int_add.outputs[0]],
        operations=[h, one, int_add],
    )

    scf_while_instr = WhileSCF(condition=condition, body=body)
    scf_while = JeffOp(
        "scf",
        "while",
        [alloc.outputs[0], counter.outputs[0]],
        [JeffValue(QubitType()), JeffValue(IntType(32))],
        scf_while_instr,
    )

    free = qubit_free(scf_while.outputs[0])

    _create_and_write_module([alloc, counter, scf_while, free], "unit_scf_while.jeff")


@register_generator
def generate_scf_do_while() -> None:
    alloc = qubit_alloc()
    counter = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 0)

    three = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 3)
    int_lt = JeffOp(
        "int",
        "ltS",
        [JeffValue(IntType(32)), three.outputs[0]],
        [JeffValue(IntType(1))],
    )
    condition = JeffRegion(
        sources=[JeffValue(QubitType()), int_lt.inputs[0]],
        targets=[int_lt.outputs[0]],
        operations=[three, int_lt],
    )

    h = quantum_gate("h", qubits=[JeffValue(QubitType())])
    one = JeffOp("int", "const32", [], [JeffValue(IntType(32))], 1)
    int_add = JeffOp(
        "int", "add", [JeffValue(IntType(32)), one.outputs[0]], [JeffValue(IntType(32))]
    )
    body = JeffRegion(
        sources=[h.inputs[0], int_add.inputs[0]],
        targets=[h.outputs[0], int_add.outputs[0]],
        operations=[h, one, int_add],
    )

    scf_do_while_instr = DoWhileSCF(body=body, condition=condition)
    scf_do_while = JeffOp(
        "scf",
        "do_while",
        [alloc.outputs[0], counter.outputs[0]],
        [JeffValue(QubitType()), JeffValue(IntType(32))],
        scf_do_while_instr,
    )

    free = qubit_free(scf_do_while.outputs[0])

    _create_and_write_module(
        [alloc, counter, scf_do_while, free],
        "unit_scf_do_while.jeff",
    )


@register_generator
def generate_bell_pair() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    h = quantum_gate("h", qubits=[alloc1.outputs[0]])
    cx = quantum_gate("x", qubits=[alloc2.outputs[0]], control_qubits=[h.outputs[0]])
    free1 = qubit_free(cx.outputs[1])
    free2 = qubit_free(cx.outputs[0])
    _create_and_write_module(
        [alloc1, alloc2, h, cx, free1, free2],
        "bell_pair.jeff",
    )


if __name__ == "__main__":
    for generator in _generators:
        generator()
