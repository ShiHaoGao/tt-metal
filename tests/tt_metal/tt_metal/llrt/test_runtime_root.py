#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise SDK-owned root resolution in relocated, fresh CPU processes."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


parser = argparse.ArgumentParser()
parser.add_argument("--probe", type=Path, required=True)
parser.add_argument("--library", type=Path, required=True)
parser.add_argument("--resources", type=Path, required=True)
parser.add_argument("--libdir", default="lib")
parser.add_argument("--libexecdir", default="libexec")
parser.add_argument("--distro-root", type=Path, default=Path("/usr/libexec/tt-metalium"))
parser.add_argument("--overlay", type=Path)
ARGS, unittest_args = parser.parse_known_args()
for name in ("libdir", "libexecdir"):
    directory = Path(getattr(ARGS, name))
    if directory.is_absolute() or ".." in directory.parts:
        parser.error(f"--{name} must be a relative install directory within the test prefix")
for name in ("probe", "library", "resources", "distro_root", "overlay"):
    value = getattr(ARGS, name)
    if value is not None:
        setattr(ARGS, name, value.resolve())


class RuntimeRoot(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="tt-runtime-root-")
        self.addCleanup(self.scratch.cleanup)
        self.base = Path(self.scratch.name)
        self.prefix = self.base / "installed"
        self.lib = self.prefix / ARGS.libdir
        self.root = self.prefix / ARGS.libexecdir / "tt-metalium"
        self.lib.mkdir(parents=True)
        shutil.copy2(ARGS.library, self.lib / ARGS.library.name)
        if ARGS.overlay:
            shutil.copy2(ARGS.overlay, self.lib / ARGS.overlay.name)
        for name in ("soc_descriptors", "core_descriptors"):
            shutil.copytree(ARGS.resources / name, self.root / "tt_metal" / name)
        self.empty_cwd = self.base / "empty-cwd"
        self.empty_cwd.mkdir()
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("TT_METAL_")}
        self.env["LD_LIBRARY_PATH"] = str(self.lib) + os.pathsep + self.env.get("LD_LIBRARY_PATH", "")

    def run_probe(self, expected, *, cwd=None, api=None):
        command = [str(ARGS.probe)]
        if api is not None:
            command += ["--api", str(api)]
        result = subprocess.run(command, cwd=cwd or self.empty_cwd, env=self.env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = next((line.removeprefix("SDK_ROOT=") for line in result.stdout.splitlines()
                       if line.startswith("SDK_ROOT=")), None)
        self.assertIsNotNone(actual, result.stdout + result.stderr)
        self.assertEqual(Path(actual), expected)

    def test_installed_library_finds_own_resources(self):
        self.run_probe(self.root)

    def test_moved_prefix_finds_moved_resources(self):
        moved = self.base / "relocated-prefix"
        self.prefix.rename(moved)
        self.env["LD_LIBRARY_PATH"] = str(moved / ARGS.libdir)
        self.run_probe(moved / ARGS.libexecdir / "tt-metalium")

    def test_environment_overrides_api_and_installed_root(self):
        env_root = self.base / "env-root"
        env_root.mkdir()
        api_root = self.base / "api-root"
        api_root.mkdir()
        self.env["TT_METAL_RUNTIME_ROOT"] = str(env_root)
        self.run_probe(env_root, api=api_root)

    def test_api_overrides_installed_root(self):
        api_root = self.base / "api-root"
        api_root.mkdir()
        self.run_probe(api_root, api=api_root)

    def test_installed_root_precedes_sdk_working_directory(self):
        checkout = self.base / "sdk-checkout"
        (checkout / "tt_metal").mkdir(parents=True)
        self.run_probe(self.root, cwd=checkout)

    def test_missing_installed_resources_retains_fallback(self):
        shutil.rmtree(self.root)
        checkout = self.base / "sdk-checkout"
        (checkout / "tt_metal").mkdir(parents=True)
        expected = ARGS.distro_root if ARGS.distro_root.is_dir() else checkout
        self.run_probe(expected, cwd=checkout)

    def test_no_root_fails_unless_distribution_install_exists(self):
        shutil.rmtree(self.root)
        if ARGS.distro_root.is_dir():
            self.run_probe(ARGS.distro_root)
        else:
            result = subprocess.run([str(ARGS.probe)], cwd=self.empty_cwd, env=self.env,
                                    text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Root Directory is not set", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main(argv=[__file__, *unittest_args])
