## An MLIR dialect for the *jeff* exchange format

This repository is contains an MLIR dialect for the [*jeff* exchange format](https://github.com/unitaryfoundation/jeff).
The purpose is to facilitate conversion to and from *jeff* for MLIR-based quantum compilers, and
reuse certain components like the jeff serialization & deserialization.

Right now, the project only contains the dialect definitions, (de)serialization is missing.
Conversion rules to and from standard MLIR dialects could also be added.

## Build instructions

The dialect sources can be incorporated directly into an existing project (in-tree), or built
separately (out-of-tree). At the moment, the CMake script supports building the dialect as a
standalone project which generates a custom `opt` tool.

To do so, make sure you have an existing MLIR build.
<details>
<summary>If you don't ...</summary>
You can get pre-built binaries from the [`munich-quantum-software/portable-mlir-toolchain`](https://github.com/munich-quantum-software/portable-mlir-toolchain) project via the installers from [`munich-quantum-software/setup-mlir`](https://github.com/munich-quantum-software/setup-mlir):

On Linux and macOS, use the following Bash command:

```bash
curl -LsSf https://github.com/munich-quantum-software/setup-mlir/releases/latest/download/setup-mlir.sh | bash -s -- -v 21.1.8 -p /path/to/installation
```

On Windows, use the following PowerShell command:

```powershell
powershell -ExecutionPolicy ByPass -c "& ([scriptblock]::Create((irm https://github.com/munich-quantum-software/setup-mlir/releases/latest/download/setup-mlir.ps1))) -llvm_version 21.1.8 -install_prefix /path/to/installation"
```

Replace `/path/to/installation` with the path to your preferred installation directory.
Then, set `MLIR_DIR` to `/path/to/installation/lib/cmake/mlir` in your CMake configuration step.

Alternatively, you can build MLIR from source following the instructions in the [MLIR documentation](https://mlir.llvm.org/getting_started/).
</details>
<br>

From here, we can build the *jeff* dialect simply via:
```sh
cmake -Bbuild -S. -GNinja && cmake --build build
```

If CMake can't find the MLIR build, it can be specified via `-DMLIR_DIR=...` during the config step,
which must point to the cmake files in the build or installation directory, for example
`/path/to/installation/lib/cmake/mlir`.

The `jeff-opt` tool will be located in the build directory under `./build/lib/opt/jeff-opt`.

TODO: Add instructions for setting up the dialect as an MLIR plugin.

## License

This *jeff* dialect is **free** and **open source**, released under the Apache License, Version 2.0.
