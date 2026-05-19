#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from analyze_real_matmul_protocol import parse_device_zones, parse_host_results, summarize_critical_stages, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Run real_matmul_protocol device-profiler cases one process at a time.")
    parser.add_argument(
        "--exe",
        type=Path,
        default=Path("build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_matmul_protocol"),
        help="Path to the real_matmul_protocol executable.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/real_matmul_protocol_cases"))
    parser.add_argument("--dims", type=int, nargs="+", default=[512, 1024], help="Square M=N dimensions to test.")
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--Ks", type=int, nargs="+", help="One or more K dimensions to test. Overrides --K.")
    parser.add_argument("--num-pages", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "profiled-cb",
            "static-input-only",
            "static-output-only",
            "static-input-output",
            "static-input-only-cbregs",
            "static-output-only-cbregs",
            "static-input-output-cbregs",
            "static-input-only-cbregs-compiletime",
            "static-output-only-cbregs-compiletime",
            "static-input-output-cbregs-compiletime",
        ],
        choices=[
            "profiled-cb",
            "static-input-only",
            "static-output-only",
            "static-input-output",
            "static-input-only-cbregs",
            "static-output-only-cbregs",
            "static-input-output-cbregs",
            "static-input-only-cbregs-compiletime",
            "static-output-only-cbregs-compiletime",
            "static-input-output-cbregs-compiletime",
        ],
    )
    parser.add_argument("--skip-check", action="store_true")
    return parser.parse_args()


def run_case(args, dim, k, mode):
    case_name = f"M{dim}_N{dim}_K{k}_S{args.num_pages}_{mode}"
    case_dir = args.out_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    host_log = case_dir / "host.log"
    device_csv = case_dir / "profile_log_device.csv"
    cache_dir = case_dir / "tt_metal_cache"

    for path in [Path("generated/profiler/.logs/profile_log_device.csv"), Path("generated/profiler/.logs/profile_log_host.csv")]:
        path.unlink(missing_ok=True)

    cmd = [
        str(args.exe),
        f"--mode={mode}",
        f"--M={dim}",
        f"--N={dim}",
        f"--K={k}",
        f"--num-pages={args.num_pages}",
        f"--repeats={args.repeats}",
        f"--device-id={args.device_id}",
    ]
    if args.skip_check:
        cmd.append("--skip-check")

    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TT_METAL_CACHE"] = str(cache_dir)
    if mode != "profiled-cb":
        env["TT_METAL_ALLOCATOR_MODE_HYBRID"] = "1"

    with host_log.open("w") as log_file:
        subprocess.run(cmd, cwd=Path.cwd(), env=env, stdout=log_file, stderr=subprocess.STDOUT, check=True)

    generated_device_csv = Path("generated/profiler/.logs/profile_log_device.csv")
    if generated_device_csv.exists():
        shutil.copy2(generated_device_csv, device_csv)
    else:
        raise RuntimeError(f"Device profiler CSV was not produced for {case_name}")

    host_rows = parse_host_results(host_log)
    zone_rows = parse_device_zones(device_csv)
    critical_rows = summarize_critical_stages(zone_rows)

    for row in host_rows:
        row["case"] = case_name
    for row in zone_rows:
        row.update({"case": case_name, "M": dim, "N": dim, "K": k, "num_pages": args.num_pages})
    for row in critical_rows:
        row.update({"case": case_name, "M": dim, "N": dim, "K": k, "num_pages": args.num_pages})

    return host_rows, zone_rows, critical_rows


def build_comparisons(critical_rows):
    grouped = defaultdict(dict)
    for row in critical_rows:
        key = (row["M"], row["N"], row["K"], row["num_pages"])
        entry = grouped[key].setdefault(row["mode"], {"critical_cycles": 0, "critical_stage": ""})
        cycles = row["median_of_core_medians"]
        if cycles is not None and cycles > entry["critical_cycles"]:
            entry["critical_cycles"] = cycles
            entry["critical_stage"] = row["stage"]

    comparisons = []
    for key, modes in sorted(grouped.items()):
        if "profiled-cb" not in modes:
            continue
        cb = modes["profiled-cb"]
        for static_mode in [
            "static-input-only",
            "static-output-only",
            "static-input-output",
            "static-input-only-cbregs",
            "static-output-only-cbregs",
            "static-input-output-cbregs",
            "static-input-only-cbregs-compiletime",
            "static-output-only-cbregs-compiletime",
            "static-input-output-cbregs-compiletime",
        ]:
            if static_mode not in modes:
                continue
            static = modes[static_mode]
            comparisons.append(
                {
                    "M": key[0],
                    "N": key[1],
                    "K": key[2],
                    "num_pages": key[3],
                    "mode": static_mode,
                    "cb_critical_stage": cb["critical_stage"],
                    "cb_critical_cycles": cb["critical_cycles"],
                    "static_critical_stage": static["critical_stage"],
                    "static_critical_cycles": static["critical_cycles"],
                    "delta_cycles_cb_minus_static": cb["critical_cycles"] - static["critical_cycles"],
                }
            )
    return comparisons


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_host_rows = []
    all_zone_rows = []
    all_critical_rows = []
    ks = args.Ks if args.Ks is not None else [args.K]
    for dim in args.dims:
        for k in ks:
            for mode in args.modes:
                print(f"running M=N={dim} K={k} pages={args.num_pages} mode={mode}", flush=True)
                host_rows, zone_rows, critical_rows = run_case(args, dim, k, mode)
                all_host_rows.extend(host_rows)
                all_zone_rows.extend(zone_rows)
                all_critical_rows.extend(critical_rows)

    comparisons = build_comparisons(all_critical_rows)
    write_csv(args.out_dir / "host_results.csv", all_host_rows)
    write_csv(args.out_dir / "zone_summary.csv", all_zone_rows)
    write_csv(args.out_dir / "critical_stage_summary.csv", all_critical_rows)
    write_csv(args.out_dir / "device_mode_comparison.csv", comparisons)

    writer = csv.DictWriter(sys.stdout, fieldnames=list(comparisons[0].keys())) if comparisons else None
    if writer:
        writer.writeheader()
        writer.writerows(comparisons)


if __name__ == "__main__":
    main()
