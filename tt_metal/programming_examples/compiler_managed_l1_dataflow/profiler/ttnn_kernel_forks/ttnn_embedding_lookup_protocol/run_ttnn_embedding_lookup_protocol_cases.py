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

from analyze_ttnn_embedding_lookup_protocol import (
    parse_device_zones,
    parse_host_results,
    summarize_critical_stages,
    summarize_host,
    write_csv,
)


def parse_shape(value):
    normalized = value.lower().replace("x", ",")
    parts = [part for part in normalized.split(",") if part]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be ROWSxVOCABxDIM, for example 1024x32000x128")
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Run TTNN embedding lookup protocol profiler cases.")
    parser.add_argument(
        "--exe",
        type=Path,
        default=Path("build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_embedding_lookup_protocol"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/ttnn_embedding_lookup_protocol_cases"))
    parser.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="+",
        default=[(256, 32000, 128), (1024, 32000, 128), (4096, 32000, 128)],
    )
    parser.add_argument("--num-pages", type=int, nargs="+", default=[2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--core-x", type=int, default=0)
    parser.add_argument("--core-y", type=int, default=0)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["cb", "static-runtime", "static-streamreg-cbregs"],
        choices=["cb", "static-runtime", "static-streamreg-cbregs"],
    )
    return parser.parse_args()


def run_case(args, rows_count, vocab_size, dim, num_pages, mode):
    case_name = f"R{rows_count}_V{vocab_size}_D{dim}_S{num_pages}_{mode}"
    case_dir = args.out_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    host_log = case_dir / "host.log"
    device_csv = case_dir / "profile_log_device.csv"
    cache_dir = case_dir / "tt_metal_cache"

    for path in [Path("generated/profiler/.logs/profile_log_device.csv"), Path("generated/profiler/.logs/profile_log_host.csv")]:
        path.unlink(missing_ok=True)

    command = [
        str(args.exe),
        f"--mode={mode}",
        f"--rows={rows_count}",
        f"--vocab-size={vocab_size}",
        f"--dim={dim}",
        f"--num-pages={num_pages}",
        f"--repeats={args.repeats}",
        f"--device-id={args.device_id}",
        f"--core-x={args.core_x}",
        f"--core-y={args.core_y}",
    ]

    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TT_METAL_CACHE"] = str(cache_dir)
    if mode != "cb":
        env["TT_METAL_ALLOCATOR_MODE_HYBRID"] = "1"

    with host_log.open("w") as log_file:
        subprocess.run(command, cwd=Path.cwd(), env=env, stdout=log_file, stderr=subprocess.STDOUT, check=True)

    generated_device_csv = Path("generated/profiler/.logs/profile_log_device.csv")
    if generated_device_csv.exists():
        shutil.copy2(generated_device_csv, device_csv)
    else:
        raise RuntimeError(f"Device profiler CSV was not produced for {case_name}")

    host_rows = parse_host_results(host_log)
    host_summary = summarize_host(host_rows)
    zone_rows = parse_device_zones(device_csv)
    critical_rows = summarize_critical_stages(zone_rows)

    common = {
        "case": case_name,
        "rows": rows_count,
        "vocab_size": vocab_size,
        "dim": dim,
        "row_bytes": dim * 2,
        "num_pages": num_pages,
        "core_grid_x": 1,
        "core_grid_y": 1,
        "max_rows_per_core": rows_count,
    }
    for row in host_rows:
        row["case"] = case_name
    for row in host_summary:
        row["case"] = case_name
    for row in zone_rows:
        row.update(common)
    for row in critical_rows:
        row.update(common)

    return host_rows, host_summary, zone_rows, critical_rows


def build_critical_comparisons(critical_rows):
    grouped = defaultdict(dict)
    for row in critical_rows:
        key = (row["rows"], row["vocab_size"], row["dim"], row["row_bytes"], row["num_pages"])
        entry = grouped[key].setdefault(row["mode"], {"critical_cycles": 0, "critical_stage": "", "core_count": 1})
        cycles = row["max_core_median"]
        if cycles is not None and cycles > entry["critical_cycles"]:
            entry["critical_cycles"] = cycles
            entry["critical_stage"] = row["stage"]
            entry["core_count"] = row["core_count"]

    comparisons = []
    for key, modes in sorted(grouped.items()):
        if "cb" not in modes:
            continue
        cb = modes["cb"]
        active_cores = max(1, cb.get("core_count", 1))
        max_rows_per_core = max(1, (key[0] + active_cores - 1) // active_cores)
        for static_mode in ["static-runtime", "static-streamreg-cbregs"]:
            if static_mode not in modes:
                continue
            static = modes[static_mode]
            delta = cb["critical_cycles"] - static["critical_cycles"]
            comparisons.append(
                {
                    "rows": key[0],
                    "vocab_size": key[1],
                    "dim": key[2],
                    "row_bytes": key[3],
                    "num_pages": key[4],
                    "core_grid_x": 1,
                    "core_grid_y": 1,
                    "active_core_count": active_cores,
                    "max_rows_per_core": max_rows_per_core,
                    "mode": static_mode,
                    "cb_critical_stage": cb["critical_stage"],
                    "cb_critical_cycles": cb["critical_cycles"],
                    "static_critical_stage": static["critical_stage"],
                    "static_critical_cycles": static["critical_cycles"],
                    "delta_cycles_cb_minus_static": delta,
                    "delta_cycles_per_row": delta / max(1, key[0]),
                    "delta_cycles_per_max_core_row": delta / max(1, max_rows_per_core),
                    "static_speedup": cb["critical_cycles"] / static["critical_cycles"]
                    if static["critical_cycles"]
                    else 0.0,
                }
            )
    return comparisons


def build_host_comparisons(host_summary_rows):
    grouped = defaultdict(dict)
    for row in host_summary_rows:
        grouped[(row["rows"], row["vocab_size"], row["dim"], row["row_bytes"], row["num_pages"])][row["mode"]] = row

    comparisons = []
    for key, modes in sorted(grouped.items()):
        if "cb" not in modes:
            continue
        cb = modes["cb"]
        for static_mode in ["static-runtime", "static-streamreg-cbregs"]:
            if static_mode not in modes:
                continue
            static = modes[static_mode]
            delta = cb["median_enqueue_finish_us"] - static["median_enqueue_finish_us"]
            comparisons.append(
                {
                    "rows": key[0],
                    "vocab_size": key[1],
                    "dim": key[2],
                    "row_bytes": key[3],
                    "num_pages": key[4],
                    "core_grid_x": 1,
                    "core_grid_y": 1,
                    "mode": static_mode,
                    "cb_median_enqueue_finish_us": cb["median_enqueue_finish_us"],
                    "static_median_enqueue_finish_us": static["median_enqueue_finish_us"],
                    "delta_us_cb_minus_static": delta,
                    "delta_us_per_row": delta / max(1, key[0]),
                    "static_speedup": cb["median_enqueue_finish_us"] / static["median_enqueue_finish_us"]
                    if static["median_enqueue_finish_us"]
                    else 0.0,
                }
            )
    return comparisons


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_host_rows = []
    all_host_summary_rows = []
    all_zone_rows = []
    all_critical_rows = []

    for rows_count, vocab_size, dim in args.shapes:
        for num_pages in args.num_pages:
            for mode in args.modes:
                print(
                    f"running rows={rows_count} vocab_size={vocab_size} dim={dim} pages={num_pages} mode={mode}",
                    flush=True,
                )
                host_rows, host_summary, zone_rows, critical_rows = run_case(
                    args, rows_count, vocab_size, dim, num_pages, mode
                )
                all_host_rows.extend(host_rows)
                all_host_summary_rows.extend(host_summary)
                all_zone_rows.extend(zone_rows)
                all_critical_rows.extend(critical_rows)

    critical_comparisons = build_critical_comparisons(all_critical_rows)
    host_comparisons = build_host_comparisons(all_host_summary_rows)
    write_csv(args.out_dir / "host_results.csv", all_host_rows)
    write_csv(args.out_dir / "host_summary.csv", all_host_summary_rows)
    write_csv(args.out_dir / "zone_summary.csv", all_zone_rows)
    write_csv(args.out_dir / "critical_stage_summary.csv", all_critical_rows)
    write_csv(args.out_dir / "device_mode_comparison.csv", critical_comparisons)
    write_csv(args.out_dir / "host_mode_comparison.csv", host_comparisons)

    writer = csv.DictWriter(sys.stdout, fieldnames=list(critical_comparisons[0].keys())) if critical_comparisons else None
    if writer:
        writer.writeheader()
        writer.writerows(critical_comparisons)


if __name__ == "__main__":
    main()
