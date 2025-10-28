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
Start by cloning the llvm-project submodule in the repo as follows (otherwise it is optional):

```sh
git submodule update --init --depth=1
```

Then, build the MLIR project for example with:

```sh
cd external/llvm-project
cmake -Bbuild -Sllvm -GNinja            \
      -DCMAKE_BUILD_TYPE=Release        \
      -DLLVM_ENABLE_PROJECTS="mlir"     \
      -DLLVM_BUILD_EXAMPLES=OFF         \
      -DLLVM_ENABLE_ASSERTIONS=ON       \
      -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
cmake --build build --target check-mlir
cd ../..
```
</details>
<br>

From here, we can build the *jeff* dialect simply via:
```sh
cmake -Bbuild -S. -GNinja && cmake --build build
```

If CMake can't find the mlir build, it can be specified via `-DMLIR_DIR=...` during the config step,
which must point to the cmake files in the build directory, for example
`./external/llvm-project/build/lib/cmake/mlir`.

The `jeff-opt` tool will be located in the build directory under `./build/lib/opt/jeff-opt`.

TODO: Add instructions for setting up the dialect as an MLIR plugin.

## License

This *jeff* dialect is **free** and **open source**, released under the Apache License, Version 2.0.
