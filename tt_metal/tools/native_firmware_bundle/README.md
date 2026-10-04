# Offline native firmware publication

`native_firmware_bundle` explicitly builds supplier firmware from the SDK
source root selected when the executable was built. It requires no device,
mesh, mock device, or `MetalContext`. Building or installing the executable
does not run it. It is not a TReX cold-compile or runtime dependency.

Run each profile into a new directory whose parent already exists:

```bash
build/tools/native_firmware_bundle --output=/path/to/firmware-disabled --profile=disabled
build/tools/native_firmware_bundle --output=/path/to/firmware-program --profile=program
```

Both flags are required. Duplicate/unknown arguments, unsupported profile
modes, existing output directories, and existing output symlinks are rejected.
The existing supplier linker requires shell-safe SDK and temporary build paths;
unsupported paths are rejected before compilation. No environment-variable
override is used to select the SDK root, cache, or profiler mode.

The tool visits every Blackhole tuple from
`enumerate_offline_compile_device_configs`. It recreates the catalog HAL with
the requested profiler allocation and uses a separate explicit `BuildEnvManager`.
Actual SDK build keys deduplicate builds; every enumerated tuple remains in the
publication record. A shared key with differing firmware recipes is rejected.
The key selects an SDK precompiled directory, not a semantic firmware identity.

The fresh directory contains:

- `tt_metal/pre-compiled/<key>/`: complete and weakened firmware artifacts from
  all actual firmware build states, including non-Tensix cores. The five Tensix
  images must satisfy the SDK's native firmware bundle admission.
- `provenance/`: actual sources and transitive compiler depfile inputs, linker
  resources, configuration catalog/YAML, source licenses, and compiler version
  output. Generated and external inputs retain separate paths. GCC system
  headers omitted by `-MMD` belong to the recorded toolchain, not this source
  snapshot. The compiler executable is recorded but is not redistributed.
- `publication.json`: all configuration tuples, actual compiler commands and
  versions, recipes, paths, native ABI records, and source snapshot inventory.
  This supplier handoff record does not replace the product's typed support
  manifest or immutable image ownership.

Compilation and copying use a private sibling staging directory and private
cache. Failed builds do not expose a public bundle. The final rename is atomic
and refuses to replace an output created concurrently. Successful publication
removes the temporary build cache.

The independent `unit_tests_native_firmware_publication` target covers argument
validation, output protection, byte preservation, atomic publication, and
cleanup. Those CPU checks do not prove compilation of the full firmware catalog
or native device deployment; receive both profile publications through the
actual SDK build before making those claims.
