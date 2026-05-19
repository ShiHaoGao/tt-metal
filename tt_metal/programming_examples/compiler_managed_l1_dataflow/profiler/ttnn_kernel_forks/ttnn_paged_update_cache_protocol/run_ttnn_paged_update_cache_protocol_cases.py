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

from analyze_ttnn_paged_update_cache_protocol import (
    parse_device_zones,
    parse_host_results,
    summarize_critical_stages,
    summarize_host,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="运行 TTNN paged_update_cache protocol profiler cases。")
    parser.add_argument(
        "--exe",
        type=Path,
        default=Path(
            "build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/"
            "ttnn_paged_update_cache_protocol"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/ttnn_paged_update_cache_protocol_cases"))
    parser.add_argument("--users", type=int, nargs="+", default=[1, 16, 32])
    parser.add_argument("--kv-heads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--head-dims", type=int, nargs="+", default=[128])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--max-seq-lens", type=int, nargs="+", default=[2048])
    parser.add_argument("--cache-idxs", type=int, nargs="+", default=[127, 1057])
    parser.add_argument("--per-user-stride", type=int, default=17)
    parser.add_argument("--num-pages", type=int, nargs="+", default=[2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--core-x", type=int, default=0)
    parser.add_argument("--core-y", type=int, default=0)
    parser.add_argument("--core-grid-x", type=int, default=0)
    parser.add_argument("--core-grid-y", type=int, default=0)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["cb", "static-runtime", "static-streamreg-cbregs"],
        choices=["cb", "static-runtime", "static-streamreg-cbregs"],
    )
    parser.add_argument("--skip-check", action="store_true")
    return parser.parse_args()


def run_case(args, users, kv_heads, head_dim, block_size, max_seq_len, cache_idx, num_pages, mode):
    case_name = (
        f"u{users}_kvh{kv_heads}_d{head_dim}_b{block_size}_s{max_seq_len}_"
        f"idx{cache_idx}_p{num_pages}"
    )
    case_dir = args.out_dir / case_name / mode
    case_dir.mkdir(parents=True, exist_ok=True)

    host_log = case_dir / "host.log"
    device_csv = case_dir / "profile_log_device.csv"
    cache_dir = case_dir / "tt_metal_cache"

    for path in [Path("generated/profiler/.logs/profile_log_device.csv"), Path("generated/profiler/.logs/profile_log_host.csv")]:
        path.unlink(missing_ok=True)

    command = [
        str(args.exe),
        f"--mode={mode}",
        f"--users={users}",
        f"--kv-heads={kv_heads}",
        f"--head-dim={head_dim}",
        f"--block-size={block_size}",
        f"--max-seq-len={max_seq_len}",
        f"--cache-idx={cache_idx}",
        f"--per-user-stride={args.per_user_stride}",
        f"--num-pages={num_pages}",
        f"--repeats={args.repeats}",
        f"--device-id={args.device_id}",
        f"--core-x={args.core_x}",
        f"--core-y={args.core_y}",
        f"--core-grid-x={args.core_grid_x}",
        f"--core-grid-y={args.core_grid_y}",
    ]
    if args.skip_check:
        command.append("--skip-check")

    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TT_METAL_ALLOCATOR_MODE_HYBRID"] = "1"
    env["TT_METAL_CACHE"] = str(cache_dir)

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

    for rows in [host_rows, host_summary, zone_rows, critical_rows]:
        for row in rows:
            row["case"] = case_name
    return host_rows, host_summary, zone_rows, critical_rows


def build_critical_comparisons(critical_rows):
    grouped = defaultdict(dict)
    for row in critical_rows:
        case = row["case"]
        entry = grouped[case].setdefault(row["mode"], {"critical_cycles": 0, "critical_stage": ""})
        cycles = row["max_core_median"]
        if cycles is not None and cycles > entry["critical_cycles"]:
            entry["critical_cycles"] = cycles
            entry["critical_stage"] = row["stage"]

    comparisons = []
    for case, modes in sorted(grouped.items()):
        if "cb" not in modes:
            continue
        cb = modes["cb"]
        for static_mode in ["static-runtime", "static-streamreg-cbregs"]:
            if static_mode not in modes:
                continue
            static = modes[static_mode]
            delta = cb["critical_cycles"] - static["critical_cycles"]
            comparisons.append(
                {
                    "case": case,
                    "mode": static_mode,
                    "cb_critical_stage": cb["critical_stage"],
                    "cb_critical_cycles": cb["critical_cycles"],
                    "static_critical_stage": static["critical_stage"],
                    "static_critical_cycles": static["critical_cycles"],
                    "delta_cycles_cb_minus_static": delta,
                    "static_speedup": cb["critical_cycles"] / static["critical_cycles"]
                    if static["critical_cycles"]
                    else 0.0,
                }
            )
    return comparisons


def build_host_comparisons(host_summary_rows):
    grouped = defaultdict(dict)
    for row in host_summary_rows:
        grouped[row["case"]][row["mode"]] = row

    comparisons = []
    for case, modes in sorted(grouped.items()):
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
                    "case": case,
                    "mode": static_mode,
                    "cb_median_enqueue_finish_us": cb["median_enqueue_finish_us"],
                    "static_median_enqueue_finish_us": static["median_enqueue_finish_us"],
                    "delta_us_cb_minus_static": delta,
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

    for users in args.users:
        for kv_heads in args.kv_heads:
            for head_dim in args.head_dims:
                for block_size in args.block_sizes:
                    for max_seq_len in args.max_seq_lens:
                        for cache_idx in args.cache_idxs:
                            for num_pages in args.num_pages:
                                for mode in args.modes:
                                    print(
                                        "running "
                                        f"users={users} kv_heads={kv_heads} head_dim={head_dim} "
                                        f"block_size={block_size} max_seq_len={max_seq_len} "
                                        f"cache_idx={cache_idx} pages={num_pages} mode={mode}",
                                        flush=True,
                                    )
                                    host_rows, host_summary, zone_rows, critical_rows = run_case(
                                        args,
                                        users,
                                        kv_heads,
                                        head_dim,
                                        block_size,
                                        max_seq_len,
                                        cache_idx,
                                        num_pages,
                                        mode,
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

    if critical_comparisons:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(critical_comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(critical_comparisons)


if __name__ == "__main__":
    main()
