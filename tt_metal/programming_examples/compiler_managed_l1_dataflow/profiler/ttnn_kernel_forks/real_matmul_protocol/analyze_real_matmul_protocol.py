#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def median(values):
    return statistics.median(values) if values else None


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize real_matmul_protocol host and device-profiler output.")
    parser.add_argument("--host-log", type=Path, required=True, help="Log captured from real_matmul_protocol stdout.")
    parser.add_argument(
        "--device-csv",
        type=Path,
        default=Path("generated/profiler/.logs/profile_log_device.csv"),
        help="TT-Metal device profiler CSV.",
    )
    parser.add_argument("--host-summary-csv", type=Path, help="Optional output CSV for host RESULT rows.")
    parser.add_argument("--zones-csv", type=Path, help="Optional output CSV for device zone summaries.")
    parser.add_argument("--critical-csv", type=Path, help="Optional output CSV for per-mode critical stage rows.")
    return parser.parse_args()


def parse_value(key, value):
    if key in {
        "repeat",
        "M",
        "N",
        "K",
        "B",
        "num_pages",
        "out_num_pages",
        "Mt",
        "Nt",
        "Kt",
        "per_core_M",
        "per_core_N",
        "out_subblock_h",
        "out_subblock_w",
        "num_blocks",
        "num_active_cores",
        "enqueue_finish_us",
    }:
        return int(value)
    if key in {"pcc", "max_abs_error"}:
        return float(value)
    if key == "ok":
        return value == "1"
    return value


def parse_host_results(path):
    results = []
    header = None
    with path.open(newline="") as log_file:
        for row in csv.reader(log_file):
            if not row:
                continue
            if row[0] == "RESULT_HEADER":
                header = row[1:]
                continue
            if row[0] != "RESULT":
                continue
            if header is None:
                raise RuntimeError(f"Encountered RESULT before RESULT_HEADER in {path}")
            if len(row) - 1 != len(header):
                raise RuntimeError(f"Unexpected RESULT row width in {path}: {row}")
            results.append({key: parse_value(key, value) for key, value in zip(header, row[1:])})
    return results


def find_header(rows):
    for index, row in enumerate(rows):
        normalized = [cell.strip() for cell in row]
        if "zone name" in normalized and "time[cycles since reset]" in normalized:
            return index, normalized
    raise RuntimeError("Could not find profiler CSV header")


def phase_is_start(value):
    return value.strip().lower() in {"zone_start", "begin", "start"}


def phase_is_end(value):
    return value.strip().lower() in {"zone_end", "end", "stop"}


def parse_zone_name(zone):
    prefix = "RMP_REUSE_"
    if not zone.startswith(prefix):
        if zone == "CBP_FW_LOCAL_CB_INIT":
            return {"mode": "firmware", "stage": "local-cb-init"}
        if zone == "CBP_FW_REMOTE_CB_INIT":
            return {"mode": "firmware", "stage": "remote-cb-init"}
        return None
    rest = zone[len(prefix) :]
    if rest.startswith("CB_"):
        mode = "profiled-cb"
        stage = rest[len("CB_") :]
    elif rest.startswith("LEVEL_C_LLK_DIRECT_FW_SKIP_CB_INIT_"):
        mode = "level-c-llk-direct-fw-skip-cb-init"
        stage = rest[len("LEVEL_C_LLK_DIRECT_FW_SKIP_CB_INIT_") :]
    elif rest.startswith("LEVEL_C_LLK_DIRECT_"):
        mode = "level-c-llk-direct"
        stage = rest[len("LEVEL_C_LLK_DIRECT_") :]
    elif rest.startswith("STATIC_INPUT_ONLY_CBREGS_COMPILETIME_"):
        mode = "static-input-only-cbregs-compiletime"
        stage = rest[len("STATIC_INPUT_ONLY_CBREGS_COMPILETIME_") :]
    elif rest.startswith("STATIC_OUTPUT_ONLY_CBREGS_COMPILETIME_"):
        mode = "static-output-only-cbregs-compiletime"
        stage = rest[len("STATIC_OUTPUT_ONLY_CBREGS_COMPILETIME_") :]
    elif rest.startswith("STATIC_INPUT_OUTPUT_CBREGS_COMPILETIME_"):
        mode = "static-input-output-cbregs-compiletime"
        stage = rest[len("STATIC_INPUT_OUTPUT_CBREGS_COMPILETIME_") :]
    elif rest.startswith("STATIC_INPUT_ONLY_CBREGS_"):
        mode = "static-input-only-cbregs"
        stage = rest[len("STATIC_INPUT_ONLY_CBREGS_") :]
    elif rest.startswith("STATIC_OUTPUT_ONLY_CBREGS_"):
        mode = "static-output-only-cbregs"
        stage = rest[len("STATIC_OUTPUT_ONLY_CBREGS_") :]
    elif rest.startswith("STATIC_INPUT_OUTPUT_CBREGS_"):
        mode = "static-input-output-cbregs"
        stage = rest[len("STATIC_INPUT_OUTPUT_CBREGS_") :]
    elif rest.startswith("STATIC_INPUT_ONLY_"):
        mode = "static-input-only"
        stage = rest[len("STATIC_INPUT_ONLY_") :]
    elif rest.startswith("STATIC_OUTPUT_ONLY_"):
        mode = "static-output-only"
        stage = rest[len("STATIC_OUTPUT_ONLY_") :]
    elif rest.startswith("STATIC_INPUT_OUTPUT_"):
        mode = "static-input-output"
        stage = rest[len("STATIC_INPUT_OUTPUT_") :]
    else:
        return None
    return {"mode": mode, "stage": stage.lower().replace("_", "-")}


def parse_device_zones(path):
    if path is None or not path.exists():
        return []

    with path.open(newline="") as csv_file:
        rows = list(csv.reader(csv_file))
    header_index, header = find_header(rows)

    time_col = header.index("time[cycles since reset]")
    zone_col = header.index("zone name")
    phase_col = header.index("type") if "type" in header else header.index("zone phase")
    core_x_col = header.index("core_x")
    core_y_col = header.index("core_y")
    risc_col = header.index("RISC processor type")

    open_zones = defaultdict(list)
    durations = defaultdict(list)
    for row in rows[header_index + 1 :]:
        if len(row) <= max(time_col, zone_col, phase_col, core_x_col, core_y_col, risc_col):
            continue
        zone = row[zone_col].strip()
        metadata = parse_zone_name(zone)
        if metadata is None:
            continue
        key = (zone, row[core_x_col].strip(), row[core_y_col].strip(), row[risc_col].strip())
        timestamp = int(row[time_col].strip())
        phase = row[phase_col].strip()
        if phase_is_start(phase):
            open_zones[key].append(timestamp)
        elif phase_is_end(phase) and open_zones[key]:
            durations[key].append(timestamp - open_zones[key].pop())

    zone_rows = []
    for key in sorted(durations):
        zone, core_x, core_y, risc = key
        samples = durations[key]
        metadata = parse_zone_name(zone)
        zone_rows.append(
            {
                "zone": zone,
                **metadata,
                "core": f"({core_x},{core_y})",
                "risc": risc,
                "count": len(samples),
                "median_cycles": median(samples),
                "min_cycles": min(samples),
                "max_cycles": max(samples),
            }
        )
    return zone_rows


def summarize_critical_stages(zone_rows):
    grouped = defaultdict(list)
    for row in zone_rows:
        grouped[(row["mode"], row["stage"])].append(row["median_cycles"])

    stage_rows = []
    for (mode, stage), values in sorted(grouped.items()):
        stage_rows.append(
            {
                "mode": mode,
                "stage": stage,
                "median_of_core_medians": median(values),
                "min_core_median": min(values),
                "max_core_median": max(values),
                "core_count": len(values),
            }
        )
    return stage_rows


def write_csv(path, rows):
    if path is None or not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_rows(title, rows, fields):
    if not rows:
        return
    print(title)
    print(",".join(fields))
    for row in rows:
        print(",".join(str(row[field]) for field in fields))
    print()


def main():
    args = parse_args()
    host_rows = parse_host_results(args.host_log)
    zone_rows = parse_device_zones(args.device_csv)
    critical_rows = summarize_critical_stages(zone_rows)

    write_csv(args.host_summary_csv, host_rows)
    write_csv(args.zones_csv, zone_rows)
    write_csv(args.critical_csv, critical_rows)

    print_rows(
        "HOST_RESULTS",
        host_rows,
        ["mode", "M", "N", "K", "num_pages", "out_num_pages", "enqueue_finish_us", "ok", "pcc", "max_abs_error"],
    )
    print_rows(
        "DEVICE_STAGE_MEDIANS",
        critical_rows,
        ["mode", "stage", "median_of_core_medians", "min_core_median", "max_core_median", "core_count"],
    )


if __name__ == "__main__":
    main()
