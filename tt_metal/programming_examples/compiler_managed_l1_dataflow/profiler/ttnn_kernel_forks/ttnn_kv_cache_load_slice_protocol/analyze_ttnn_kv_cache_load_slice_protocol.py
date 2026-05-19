#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


MODE_LABELS = {
    "CB": "cb",
    "STATIC_RUNTIME": "static-runtime",
    "STATIC_STREAMREG_CBREGS": "static-streamreg-cbregs",
}

STAGE_LABELS = {
    "READER": "reader",
    "WRITER": "writer",
}

ZONE_RE = re.compile(r"^TKVL_(?P<mode>CB|STATIC_RUNTIME|STATIC_STREAMREG_CBREGS)_(?P<stage>READER|WRITER)$")
CASE_RE = re.compile(
    r"ttnn_kv_cache_load_slice_protocol: input_seq_tiles=(?P<input_seq_tiles>\d+), "
    r"output_seq_tiles=(?P<output_seq_tiles>\d+), head_dim_tiles=(?P<head_dim_tiles>\d+), "
    r"start_seq_tile=(?P<start_seq_tile>\d+), output_tiles=(?P<output_tiles>\d+), "
    r"tile_size=(?P<tile_size>\d+), repeats=(?P<repeats>\d+), "
    r"core=\((?P<core_x>\d+), (?P<core_y>\d+)\), "
    r"core_grid=\((?P<core_grid_x>\d+), (?P<core_grid_y>\d+)\)"
)
RESULT_RE = re.compile(
    r"mode=(?P<mode>\S+)\s+repeat=(?P<repeat>\d+) "
    r"enqueue_finish_us=(?P<enqueue_finish_us>\d+) "
    r"max_abs_error=(?P<max_abs_error>[-+0-9.eE]+) (?P<status>ok|FAILED)"
)


def median(values):
    return statistics.median(values) if values else None


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize TTNN KV-cache load-slice protocol profiler output.")
    parser.add_argument("--host-log", type=Path, required=True)
    parser.add_argument("--device-csv", type=Path, default=Path("generated/profiler/.logs/profile_log_device.csv"))
    parser.add_argument("--host-results-csv", type=Path)
    parser.add_argument("--host-summary-csv", type=Path)
    parser.add_argument("--zones-csv", type=Path)
    parser.add_argument("--critical-csv", type=Path)
    return parser.parse_args()


def parse_host_results(path):
    metadata = {}
    rows = []
    with path.open() as log_file:
        for line in log_file:
            case_match = CASE_RE.search(line)
            if case_match:
                metadata = {
                    "input_seq_tiles": int(case_match.group("input_seq_tiles")),
                    "output_seq_tiles": int(case_match.group("output_seq_tiles")),
                    "head_dim_tiles": int(case_match.group("head_dim_tiles")),
                    "start_seq_tile": int(case_match.group("start_seq_tile")),
                    "output_tiles": int(case_match.group("output_tiles")),
                    "tile_size": int(case_match.group("tile_size")),
                    "core_x": int(case_match.group("core_x")),
                    "core_y": int(case_match.group("core_y")),
                    "core_grid_x": int(case_match.group("core_grid_x")),
                    "core_grid_y": int(case_match.group("core_grid_y")),
                }
                continue
            result_match = RESULT_RE.search(line)
            if not result_match:
                continue
            if not metadata:
                raise RuntimeError(f"Encountered result row before case metadata in {path}")
            rows.append(
                {
                    **metadata,
                    "mode": result_match.group("mode"),
                    "repeat": int(result_match.group("repeat")),
                    "enqueue_finish_us": int(result_match.group("enqueue_finish_us")),
                    "max_abs_error": float(result_match.group("max_abs_error")),
                    "ok": result_match.group("status") == "ok",
                }
            )
    return rows


def summarize_host(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["input_seq_tiles"],
            row["output_seq_tiles"],
            row["head_dim_tiles"],
            row["start_seq_tile"],
            row["core_grid_x"],
            row["core_grid_y"],
            row["mode"],
        )
        grouped[key].append(row)
    summaries = []
    for key, samples in sorted(grouped.items()):
        output_tiles = samples[0]["output_tiles"]
        enqueue_values = [sample["enqueue_finish_us"] for sample in samples]
        summaries.append(
            {
                "input_seq_tiles": key[0],
                "output_seq_tiles": key[1],
                "head_dim_tiles": key[2],
                "start_seq_tile": key[3],
                "output_tiles": output_tiles,
                "core_grid_x": key[4],
                "core_grid_y": key[5],
                "mode": key[6],
                "count": len(samples),
                "ok_all": all(sample["ok"] for sample in samples),
                "median_enqueue_finish_us": median(enqueue_values),
                "min_enqueue_finish_us": min(enqueue_values),
                "max_enqueue_finish_us": max(enqueue_values),
                "max_abs_error": max(sample["max_abs_error"] for sample in samples),
                "median_us_per_output_tile": median(enqueue_values) / max(1, output_tiles),
            }
        )
    return summaries


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
    match = ZONE_RE.match(zone)
    if not match:
        return None
    return {
        "mode": MODE_LABELS[match.group("mode")],
        "stage": STAGE_LABELS[match.group("stage")],
    }


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


def main():
    args = parse_args()
    host_rows = parse_host_results(args.host_log)
    host_summary = summarize_host(host_rows)
    zone_rows = parse_device_zones(args.device_csv)
    critical_rows = summarize_critical_stages(zone_rows)
    write_csv(args.host_results_csv, host_rows)
    write_csv(args.host_summary_csv, host_summary)
    write_csv(args.zones_csv, zone_rows)
    write_csv(args.critical_csv, critical_rows)
    for row in critical_rows:
        print(row)


if __name__ == "__main__":
    main()
