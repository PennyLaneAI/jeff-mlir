from pathlib import Path
from jeff import (
    JeffRegion,
    JeffModule,
    FunctionDef,
    qubit_alloc,
    qubit_free,
    quantum_gate,
)


def generate_bell_pair() -> None:
    alloc1 = qubit_alloc()
    alloc2 = qubit_alloc()
    h = quantum_gate("h", qubits=[alloc1.outputs[0]])
    cx = quantum_gate("x", qubits=[alloc2.outputs[0]], control_qubits=[h.outputs[0]])
    free1 = qubit_free(cx.outputs[1])
    free2 = qubit_free(cx.outputs[0])

    body = JeffRegion(
        sources=[],
        targets=[],
        operations=[alloc1, alloc2, h, cx, free1, free2],
    )
    function = FunctionDef(name="main", body=body)
    module = JeffModule([function])

    output_file = Path(__file__).parent / "bell_pair.jeff"
    output_file.unlink(missing_ok=True)
    module.write_out(output_file)


if __name__ == "__main__":
    generate_bell_pair()
