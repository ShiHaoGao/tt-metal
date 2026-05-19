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
from pathlib import Path


TARGETED_DEVICE_CASES = [
    (
        "single_m1_n4_k1_s2",
        [
            "--op=matmul-single",
            "--matmul-m-tiles=1",
            "--matmul-n-tiles=4",
            "--matmul-k-tiles=1",
            "--num-slots=2",
        ],
    ),
    (
        "single_m1_n4_k4_s2",
        [
            "--op=matmul-single",
            "--matmul-m-tiles=1",
            "--matmul-n-tiles=4",
            "--matmul-k-tiles=4",
            "--num-slots=2",
        ],
    ),
    (
        "single_m1_n4_k8_s2",
        [
            "--op=matmul-single",
            "--matmul-m-tiles=1",
            "--matmul-n-tiles=4",
            "--matmul-k-tiles=8",
            "--num-slots=2",
        ],
    ),
    (
        "single_m1_n4_k16_s2",
        [
            "--op=matmul-single",
            "--matmul-m-tiles=1",
            "--matmul-n-tiles=4",
            "--matmul-k-tiles=16",
            "--num-slots=2",
        ],
    ),
    (
        "single_m1_n16_k8_s1",
        [
            "--op=matmul-single",
            "--matmul-m-tiles=1",
            "--matmul-n-tiles=16",
            "--matmul-k-tiles=8",
            "--num-slots=1",
        ],
    ),
    (
        "single_m1_n16_k8_s4",
        [
            "--op=matmul-single",
            "--matmul-m-tiles=1",
            "--matmul-n-tiles=16",
            "--matmul-k-tiles=8",
            "--num-slots=4",
        ],
    ),
    (
        "single_m4_n4_k4_s2",
        [
            "--op=matmul-single",
            "--matmul-m-tiles=4",
            "--matmul-n-tiles=4",
            "--matmul-k-tiles=4",
            "--num-slots=2",
        ],
    ),
    (
        "block_g1x2_m2_n4_k8_s2",
        [
            "--op=matmul-block",
            "--core-grid-x=1",
            "--core-grid-y=2",
            "--matmul-m-tiles=2",
            "--matmul-n-tiles=4",
            "--matmul-k-tiles=8",
            "--num-slots=2",
        ],
    ),
    (
        "block_g2x2_m2_n8_k8_s2",
        [
            "--op=matmul-block",
            "--core-grid-x=2",
            "--core-grid-y=2",
            "--matmul-m-tiles=2",
            "--matmul-n-tiles=8",
            "--matmul-k-tiles=8",
            "--num-slots=2",
        ],
    ),
]

MEMORY_BOUND_CASES = [
    (
        "tile_add_t256_s2",
        [
            "--op=tile-add",
            "--tiles=256",
            "--num-slots=2",
        ],
    ),
    (
        "tile_add_t1024_s2",
        [
            "--op=tile-add",
            "--tiles=1024",
            "--num-slots=2",
        ],
    ),
    (
        "tile_add_t4096_s2",
        [
            "--op=tile-add",
            "--tiles=4096",
            "--num-slots=2",
        ],
    ),
    (
        "tile_add_t1024_s4",
        [
            "--op=tile-add",
            "--tiles=1024",
            "--num-slots=4",
        ],
    ),
    (
        "tile_add_t4096_s4",
        [
            "--op=tile-add",
            "--tiles=4096",
            "--num-slots=4",
        ],
    ),
    (
        "eltwise_chain_t1024_c4_s2",
        [
            "--op=eltwise-chain",
            "--tiles=1024",
            "--chain-depth=4",
            "--num-slots=2",
        ],
    ),
    (
        "eltwise_chain_t1024_c16_s2",
        [
            "--op=eltwise-chain",
            "--tiles=1024",
            "--chain-depth=16",
            "--num-slots=2",
        ],
    ),
    (
        "eltwise_chain_t4096_c4_s2",
        [
            "--op=eltwise-chain",
            "--tiles=4096",
            "--chain-depth=4",
            "--num-slots=2",
        ],
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run per-case static protocol modeling profiles.")
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/static_protocol_modeling"),
        help="Path to the static_protocol_modeling executable.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/static_protocol_modeling_device_cases"),
        help="Directory for logs and aggregated CSVs.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per mode for each case.")
    parser.add_argument("--mode", default="all", help="Benchmark mode selection.")
    parser.add_argument(
        "--case-set",
        default="targeted-device",
        choices=["targeted-device", "memory-bound"],
        help="Named case set to run.",
    )
    parser.add_argument("--case-filter", default="", help="Run only cases whose label contains this substring.")
    return parser.parse_args()


def cases_for_case_set(case_set):
    if case_set == "targeted-device":
        return TARGETED_DEVICE_CASES
    if case_set == "memory-bound":
        return MEMORY_BOUND_CASES
    raise ValueError(f"Unknown case set: {case_set}")


def modes_for_selection(mode):
    if mode == "all":
        return ["cb", "static-runtime", "static-compiletime", "static-serialized", "static-streamreg-cbregs"]
    return [mode]


def run_logged(command, env, log_path, cwd):
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def int_field(row, field):
    return int(float(row[field]))


def float_field(row, field):
    return float(row[field])


def build_critical_summary(zone_rows):
    grouped = {}
    key_fields = [
        "op",
        "tiles",
        "num_slots",
        "chain_depth",
        "matmul_m_tiles",
        "matmul_n_tiles",
        "matmul_k_tiles",
        "core_grid_x",
        "core_grid_y",
        "mode",
    ]
    metadata_fields = key_fields + ["local_output_tiles", "local_input_tile_pairs"]
    for row in zone_rows:
        key = tuple(row[field] for field in key_fields)
        cycles = float_field(row, "median_cycles")
        if key not in grouped or cycles > float_field(grouped[key], "critical_cycles"):
            grouped[key] = {
                **{field: row[field] for field in metadata_fields},
                "critical_cycles": cycles,
                "critical_stage": row["stage"],
                "critical_core": row["core"],
                "critical_risc": row["risc"],
            }
    return sorted(
        grouped.values(),
        key=lambda row: (
            row["op"],
            int_field(row, "matmul_m_tiles"),
            int_field(row, "matmul_n_tiles"),
            int_field(row, "matmul_k_tiles"),
            int_field(row, "num_slots"),
            int_field(row, "core_grid_x"),
            int_field(row, "core_grid_y"),
            row["mode"],
        ),
    )


def build_critical_comparison(critical_rows):
    case_fields = [
        "op",
        "tiles",
        "num_slots",
        "chain_depth",
        "matmul_m_tiles",
        "matmul_n_tiles",
        "matmul_k_tiles",
        "core_grid_x",
        "core_grid_y",
    ]
    grouped = {}
    for row in critical_rows:
        key = tuple(row[field] for field in case_fields)
        grouped.setdefault(key, {})[row["mode"]] = row

    rows = []
    for key, modes in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            int(item[0][4]),
            int(item[0][5]),
            int(item[0][6]),
            int(item[0][2]),
            int(item[0][7]),
            int(item[0][8]),
        ),
    ):
        if "cb" not in modes or "static-runtime" not in modes:
            continue
        cb = modes["cb"]
        runtime = modes["static-runtime"]
        compiletime = modes.get("static-compiletime")
        serialized = modes.get("static-serialized")
        cbregs = modes.get("static-streamreg-cbregs")
        local_output_tiles = max(1, int_field(cb, "local_output_tiles"))
        local_input_tile_pairs = max(1, int_field(cb, "local_input_tile_pairs"))
        delta = float_field(cb, "critical_cycles") - float_field(runtime, "critical_cycles")
        row = {
            "op": key[0],
            "tiles": key[1],
            "num_slots": key[2],
            "chain_depth": key[3],
            "matmul_m_tiles": key[4],
            "matmul_n_tiles": key[5],
            "matmul_k_tiles": key[6],
            "core_grid_x": key[7],
            "core_grid_y": key[8],
            "local_output_tiles": local_output_tiles,
            "local_input_tile_pairs": local_input_tile_pairs,
            "cb_critical_cycles": float_field(cb, "critical_cycles"),
            "static_runtime_critical_cycles": float_field(runtime, "critical_cycles"),
            "delta_cb_static_runtime_cycles": delta,
            "delta_cycles_per_local_output_tile": delta / local_output_tiles,
            "delta_cycles_per_local_input_pair": delta / local_input_tile_pairs,
            "static_runtime_speedup": float_field(cb, "critical_cycles") / float_field(runtime, "critical_cycles"),
            "critical_stage": cb["critical_stage"],
        }
        if compiletime is not None:
            row["delta_static_runtime_compiletime_cycles"] = float_field(runtime, "critical_cycles") - float_field(
                compiletime, "critical_cycles"
            )
        if serialized is not None:
            row["static_serialized_minus_runtime_cycles"] = float_field(serialized, "critical_cycles") - float_field(
                runtime, "critical_cycles"
            )
        if cbregs is not None:
            cbregs_cycles = float_field(cbregs, "critical_cycles")
            row["static_streamreg_cbregs_critical_cycles"] = cbregs_cycles
            row["delta_cb_static_streamreg_cbregs_cycles"] = float_field(cb, "critical_cycles") - cbregs_cycles
            row["delta_cbregs_runtime_cycles"] = cbregs_cycles - float_field(runtime, "critical_cycles")
            row["static_streamreg_cbregs_speedup"] = (
                float_field(cb, "critical_cycles") / cbregs_cycles if cbregs_cycles else 0.0
            )
        rows.append(row)
    return rows


def main():
    args = parse_args()
    repo_root = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    analyzer = script_dir / "analyze_static_protocol.py"
    executable = args.executable if args.executable.is_absolute() else repo_root / args.executable
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_host_rows = []
    all_zone_rows = []
    all_device_comparison_rows = []
    selected_modes = modes_for_selection(args.mode)

    selected_cases = cases_for_case_set(args.case_set)
    if args.case_filter:
        selected_cases = [(label, case_args) for label, case_args in selected_cases if args.case_filter in label]
        if not selected_cases:
            raise ValueError(f"No cases matched --case-filter={args.case_filter!r}")

    for label, case_args in selected_cases:
        print(f"=== {label} ===", flush=True)
        log_path = out_dir / f"{label}.log"
        device_csv = out_dir / f"{label}_profile_log_device.csv"
        summary_csv = out_dir / f"{label}_summary.csv"
        zones_csv = out_dir / f"{label}_zones.csv"
        device_comparison_csv = out_dir / f"{label}_device_comparison.csv"
        analysis_log = out_dir / f"{label}_analysis.txt"
        case_host_rows = []
        case_zone_rows = []
        case_device_comparison_rows = []

        for mode in selected_modes:
            mode_label = mode.replace("-", "_")
            mode_log_path = out_dir / f"{label}_{mode_label}.log"
            mode_device_csv = out_dir / f"{label}_{mode_label}_profile_log_device.csv"
            mode_summary_csv = out_dir / f"{label}_{mode_label}_summary.csv"
            mode_zones_csv = out_dir / f"{label}_{mode_label}_zones.csv"
            mode_device_comparison_csv = out_dir / f"{label}_{mode_label}_device_comparison.csv"
            mode_analysis_log = out_dir / f"{label}_{mode_label}_analysis.txt"

            env = os.environ.copy()
            env["TT_METAL_DEVICE_PROFILER"] = "1"
            env["TT_METAL_CACHE"] = str(out_dir / f"cache_{label}_{mode_label}")
            env.pop("SPM_PROFILE_CASE_LABELS", None)

            command = [
                str(executable),
                f"--mode={mode}",
                f"--repeats={args.repeats}",
                *case_args,
            ]
            return_code = run_logged(command, env, mode_log_path, repo_root)
            if return_code != 0:
                return return_code

            generated_device_csv = repo_root / "generated/profiler/.logs/profile_log_device.csv"
            if generated_device_csv.exists():
                shutil.copy2(generated_device_csv, mode_device_csv)
                shutil.copy2(generated_device_csv, device_csv)

            analyze_command = [
                sys.executable,
                str(analyzer),
                "--host-log",
                str(mode_log_path),
                "--device-csv",
                str(mode_device_csv),
                "--drop-first-repeat",
                "--summary-csv",
                str(mode_summary_csv),
                "--zones-csv",
                str(mode_zones_csv),
                "--device-comparison-csv",
                str(mode_device_comparison_csv),
            ]
            with mode_analysis_log.open("w") as output:
                subprocess.run(analyze_command, cwd=repo_root, stdout=output, stderr=subprocess.STDOUT, check=True)

            case_host_rows.extend(read_csv_rows(mode_summary_csv))
            case_zone_rows.extend(read_csv_rows(mode_zones_csv))
            case_device_comparison_rows.extend(read_csv_rows(mode_device_comparison_csv))

        with log_path.open("w") as combined_log:
            for mode in selected_modes:
                mode_label = mode.replace("-", "_")
                mode_log_path = out_dir / f"{label}_{mode_label}.log"
                if mode_log_path.exists():
                    combined_log.write(f"=== mode={mode} ===\n")
                    combined_log.write(mode_log_path.read_text())
                    combined_log.write("\n")

        write_csv(summary_csv, case_host_rows)
        write_csv(zones_csv, case_zone_rows)
        write_csv(device_comparison_csv, case_device_comparison_rows)
        with analysis_log.open("w") as combined_analysis_log:
            for mode in selected_modes:
                mode_label = mode.replace("-", "_")
                mode_analysis_log = out_dir / f"{label}_{mode_label}_analysis.txt"
                if mode_analysis_log.exists():
                    combined_analysis_log.write(f"=== mode={mode} ===\n")
                    combined_analysis_log.write(mode_analysis_log.read_text())
                    combined_analysis_log.write("\n")

        all_host_rows.extend(case_host_rows)
        all_zone_rows.extend(case_zone_rows)
        all_device_comparison_rows.extend(case_device_comparison_rows)

    write_csv(out_dir / "combined_host_summary.csv", all_host_rows)
    write_csv(out_dir / "combined_device_zones.csv", all_zone_rows)
    write_csv(out_dir / "combined_device_comparison.csv", all_device_comparison_rows)
    critical_summary = build_critical_summary(all_zone_rows)
    critical_comparison = build_critical_comparison(critical_summary)
    write_csv(out_dir / "combined_device_critical_summary.csv", critical_summary)
    write_csv(out_dir / "combined_device_critical_comparison.csv", critical_comparison)
    print(f"combined_host_summary={out_dir / 'combined_host_summary.csv'}")
    print(f"combined_device_zones={out_dir / 'combined_device_zones.csv'}")
    print(f"combined_device_comparison={out_dir / 'combined_device_comparison.csv'}")
    print(f"combined_device_critical_summary={out_dir / 'combined_device_critical_summary.csv'}")
    print(f"combined_device_critical_comparison={out_dir / 'combined_device_critical_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
