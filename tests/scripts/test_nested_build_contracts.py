# SPDX-License-Identifier: Apache-2.0
"""Run nested source and installed loader contracts without cards or downloads.

The Git fixture mirrors the declared HTTPS UMD endpoint locally and rejects
SSH. The CMake fixture executes the complete production tt_metal/CMakeLists.txt
with tiny dependency targets, then installs and loads its library. It does not
build the SDK or claim that the real MPI implementation has been exercised.

Run directly with ``python tests/scripts/test_nested_build_contracts.py``.
"""

import configparser
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class NestedBuildContractsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="metalium-nested-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = dict(os.environ)
        for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "GIT_CONFIG_PARAMETERS",
                     "GIT_CONFIG_COUNT", "GIT_DIR", "GIT_WORK_TREE"):
            self.env.pop(name, None)
        self.env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                        GIT_TERMINAL_PROMPT="0")

    def checked(self, *command, cwd=None, env=None):
        result = subprocess.run(list(map(str, command)), cwd=cwd, env=env or self.env,
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_recursive_umd_acquisition_needs_no_ssh(self):
        config = configparser.ConfigParser()
        config.read(ROOT / ".gitmodules")
        url = config['submodule "tt_metal/third_party/umd"']["url"]
        mirror = self.root / "umd-mirror"
        mirror.mkdir()
        self.checked("git", "init", "--quiet", mirror)
        (mirror / "CMakeLists.txt").write_text("project(umd LANGUAGES NONE)\n")
        self.checked("git", "add", "CMakeLists.txt", cwd=mirror)
        self.checked("git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                     "commit", "--quiet", "-m", "UMD fixture", cwd=mirror)
        revision = self.checked("git", "rev-parse", "HEAD", cwd=mirror).strip()

        parent = self.root / "metalium-source"
        parent.mkdir()
        self.checked("git", "init", "--quiet", parent)
        (parent / ".gitmodules").write_text(
            '[submodule "tt_metal/third_party/umd"]\n'
            '\tpath = tt_metal/third_party/umd\n'
            f'\turl = {url}\n'
        )
        self.checked("git", "add", ".gitmodules", cwd=parent)
        self.checked("git", "update-index", "--add", "--cacheinfo",
                     f"160000,{revision},tt_metal/third_party/umd", cwd=parent)
        self.checked("git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                     "commit", "--quiet", "-m", "Metalium fixture", cwd=parent)
        reject_ssh = self.root / "reject-ssh"
        reject_ssh.write_text("#!/bin/sh\necho 'unexpected SSH source acquisition' >&2\nexit 41\n")
        reject_ssh.chmod(0o755)
        env = dict(self.env, GIT_SSH_COMMAND=str(reject_ssh), GIT_SSH_VARIANT="ssh")
        self.checked("git", "-c", "protocol.file.allow=always", "-c",
                     f"url.{mirror.as_uri()}.insteadOf=https://github.com/Epiler-project/tt-umd.git",
                     "submodule", "update", "--init", "--recursive", cwd=parent, env=env)
        self.assertEqual(self.checked("git", "rev-parse", "HEAD",
                                      cwd=parent / "tt_metal/third_party/umd").strip(), revision)
        self.assertEqual((parent / "tt_metal/third_party/umd/CMakeLists.txt").read_text(),
                         "project(umd LANGUAGES NONE)\n")

    def cmake_fixture(self, *, mpi_kind, enabled):
        source = self.root / "source"
        metal = source / "tt_metal"
        metal.mkdir(parents=True)
        shutil.copy2(ROOT / "tt_metal/CMakeLists.txt", metal / "CMakeLists.txt")
        (metal / "sources.cmake").write_text('set(TT_METAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/probe.cpp")\n')
        (metal / "probe.cpp").write_text(
            'extern "C" int distributed_probe();\n'
            'extern "C" int metalium_probe() { return distributed_probe(); }\n'
        )
        modules = source / "cmake"
        modules.mkdir()
        (modules / "flatbuffers.cmake").write_text("function(GENERATE_FBS_HEADER)\nendfunction()\n")
        for directory in ("logging", "hw", "hostdevcommon", "common", "jit_build", "llrt",
                          "tools", "impl", "detail", "fabric", "impl/experimental/udm",
                          "impl/experimental/noc_estimator"):
            child = metal / directory
            child.mkdir(parents=True, exist_ok=True)
            (child / "CMakeLists.txt").write_text("")

        external = self.root / "external mpi" / "lib"
        external.mkdir(parents=True)
        mpi_source = self.root / "mpi.cpp"
        mpi_source.write_text('extern "C" int fixture_mpi() { return 7; }\n')
        mpi_library = external / "libfixture_mpi.so"
        self.checked("c++", "-shared", "-fPIC", mpi_source, "-Wl,-soname,libfixture_mpi.so",
                     "-o", mpi_library)
        if mpi_kind == "shared":
            mpi_targets = (
                'add_library(OpenMPI::MPI SHARED IMPORTED GLOBAL)\n'
                f'set_target_properties(OpenMPI::MPI PROPERTIES IMPORTED_LOCATION "{mpi_library}")\n'
            )
        else:
            mpi_targets = (
                'add_library(MPI::MPI_C INTERFACE IMPORTED GLOBAL)\n'
                f'set_target_properties(MPI::MPI_C PROPERTIES INTERFACE_LINK_LIBRARIES "{mpi_library}")\n'
                'add_library(OpenMPI::MPI ALIAS MPI::MPI_C)\n'
            )
        distributed = metal / "distributed"
        distributed.mkdir()
        (distributed / "probe.cpp").write_text(
            '#ifdef FIXTURE_MPI_ENABLED\nextern "C" int fixture_mpi();\n#endif\n'
            'extern "C" int distributed_probe() {\n#ifdef FIXTURE_MPI_ENABLED\n'
            'return fixture_mpi();\n#else\nreturn 7;\n#endif\n}\n'
        )
        (distributed / "CMakeLists.txt").write_text(
            mpi_targets + 'add_library(distributed OBJECT probe.cpp)\n'
            'if(ENABLE_DISTRIBUTED)\n'
            '  target_link_libraries(distributed PRIVATE OpenMPI::MPI)\n'
            '  target_compile_definitions(distributed PRIVATE FIXTURE_MPI_ENABLED)\n'
            'endif()\n'
        )
        (source / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.25)
project(MetaliumRpathFixture LANGUAGES C CXX)
include(GNUInstallDirs)
set(BUILD_SHARED_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(TT_INSTALL OFF)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
function(tt_reuse_precompile_headers)
endfunction()
add_custom_target(hw_toolchain)
add_library(fixture_dependency INTERFACE)
foreach(name IN ITEMS umd::tt-umd enchantum::enchantum fmt::fmt-header-only
    nlohmann_json::nlohmann_json TT::Metalium::HostDevCommon TT::STL tt-logger::tt-logger
    Metalium::Metal::Impl FlatBuffers::FlatBuffers TT::Metalium::Logging
    TT::ScaleoutTools Reflect::Reflect)
  add_library(${name} ALIAS fixture_dependency)
endforeach()
foreach(name IN ITEMS TracyClient metal_common_libs common jit_build llrt detail fabric
    tt_metal_noc_estimator)
  add_library(${name} INTERFACE)
endforeach()
add_subdirectory(tt_metal)
install(TARGETS tt_metal LIBRARY DESTINATION lib)
''')
        build = self.root / "build"
        prefix = self.root / "installed"
        self.checked("cmake", "-S", source, "-B", build, "-G", "Ninja",
                     f"-DENABLE_DISTRIBUTED={'ON' if enabled else 'OFF'}")
        self.checked("cmake", "--build", build, "--parallel", str(os.cpu_count() or 1))
        self.checked("cmake", "--install", build, "--prefix", prefix)
        library = prefix / "lib/libtt_metal.so"
        dynamic = self.checked("readelf", "-d", library)
        runpath = next(line.split("[", 1)[1].split("]", 1)[0]
                       for line in dynamic.splitlines() if "(RUNPATH)" in line)
        self.checked(sys.executable, "-c",
                     "import ctypes, sys; assert ctypes.CDLL(sys.argv[1]).metalium_probe() == 7",
                     library, cwd=self.root)
        return runpath.split(":"), str(external)

    def test_enabled_shared_mpi_loads_after_install(self):
        runpath, mpi_directory = self.cmake_fixture(mpi_kind="shared", enabled=True)
        self.assertIn(mpi_directory, runpath)
        self.assertIn("$ORIGIN", runpath)

    def test_enabled_interface_mpi_loads_after_install(self):
        runpath, mpi_directory = self.cmake_fixture(mpi_kind="interface", enabled=True)
        self.assertIn(mpi_directory, runpath)
        self.assertIn("$ORIGIN", runpath)

    def test_disabled_shared_mpi_keeps_only_origin(self):
        runpath, _ = self.cmake_fixture(mpi_kind="shared", enabled=False)
        self.assertEqual(runpath, ["$ORIGIN"])

    def test_disabled_interface_mpi_keeps_only_origin(self):
        runpath, _ = self.cmake_fixture(mpi_kind="interface", enabled=False)
        self.assertEqual(runpath, ["$ORIGIN"])


if __name__ == "__main__":
    unittest.main()
