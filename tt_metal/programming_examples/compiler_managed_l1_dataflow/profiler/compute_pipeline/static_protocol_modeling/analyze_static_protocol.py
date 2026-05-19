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
    "STATIC_COMPILETIME": "static-compiletime",
    "STATIC_SERIALIZED": "static-serialized",
    "STATIC_STREAMREG": "static-streamreg",
    "STATIC_STREAMREG_CBREGS": "static-streamreg-cbregs",
    "STATIC_STREAMREG_CBREGS_COMPILETIME": "static-streamreg-cbregs-compiletime",
}

OP_LABELS = {
    "TILE_ADD": "tile-add",
    "ELTWISE_CHAIN": "eltwise-chain",
    "MATMUL_SINGLE": "matmul-single",
    "MATMUL_BLOCK": "matmul-block",
}

STAGE_LABELS = {
    "READER": "reader",
    "WRITER": "writer",
    "COMPUTE_UNPACK": "compute-unpack",
    "COMPUTE_MATH": "compute-math",
    "COMPUTE_PACK": "compute-pack",
}

MODE_PATTERN = (
    "CB|STATIC_RUNTIME|STATIC_COMPILETIME|STATIC_SERIALIZED|STATIC_STREAMREG|"
    "STATIC_STREAMREG_CBREGS|STATIC_STREAMREG_CBREGS_COMPILETIME"
)
STAGE_PATTERN = "READER|WRITER|COMPUTE_UNPACK|COMPUTE_MATH|COMPUTE_PACK"

TILE_ZONE_RE = re.compile(
    rf"^SPM_(?P<op>TILE_ADD|ELTWISE_CHAIN)_T(?P<tiles>\d+)_C(?P<chain_depth>\d+)_S(?P<num_slots>\d+)_"
    rf"(?P<mode>{MODE_PATTERN})_(?P<stage>{STAGE_PATTERN})$"
)
MATMUL_ZONE_RE = re.compile(
    rf"^SPM_(?P<op>MATMUL_SINGLE|MATMUL_BLOCK)_M(?P<matmul_m_tiles>\d+)_N(?P<matmul_n_tiles>\d+)"
    rf"_K(?P<matmul_k_tiles>\d+)_S(?P<num_slots>\d+)_G(?P<core_grid_x>\d+)X(?P<core_grid_y>\d+)_"
    rf"(?P<mode>{MODE_PATTERN})_(?P<stage>{STAGE_PATTERN})$"
)
OLD_ZONE_RE = re.compile(rf"^SPM_(?P<op>TILE_ADD|ELTWISE_CHAIN|MATMUL_SINGLE|MATMUL_BLOCK)_(?P<mode>{MODE_PATTERN})_(?P<stage>{STAGE_PATTERN})$")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize static protocol modeling host and device results.")
    parser.add_argument("--host-log", type=Path, help="Log captured from static_protocol_modeling stdout.")
    parser.add_argument(
        "--device-csv",
        type=Path,
        default=Path("generated/profiler/.logs/profile_log_device.csv"),
        help="TT-Metal device profiler CSV.",
    )
    parser.add_argument("--summary-csv", type=Path, help="Optional CSV path for host summary rows.")
    parser.add_argument("--zones-csv", type=Path, help="Optional CSV path for device zone summary rows.")
    parser.add_argument("--device-comparison-csv", type=Path, help="Optional CSV path for device mode comparison rows.")
    parser.add_argument(
        "--drop-first-repeat",
        action="store_true",
        help="Drop the lowest repeat index in each host-result group to avoid JIT/first-run enqueue noise.",
    )
    return parser.parse_args()


def median(values):
    return statistics.median(values) if values else None


def parse_int(value):
    return int(value)


def parse_float(value):
    return float(value)


def parse_host_results(path):
    if path is None:
        return []

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
                raise RuntimeError("Encountered RESULT row before RESULT_HEADER")
            if len(row) - 1 != len(header):
                raise RuntimeError(f"Unexpected RESULT row width in {path}: {row}")

            record = {}
            for key, value in zip(header, row[1:]):
                record[key] = value

            numeric_fields = {
                "repeat": parse_int,
                "tiles": parse_int,
                "num_slots": parse_int,
                "slot_bytes": parse_int,
                "chain_depth": parse_int,
                "matmul_m_tiles": parse_int,
                "matmul_n_tiles": parse_int,
                "matmul_k_tiles": parse_int,
                "input_tile_pairs": parse_int,
                "output_tiles": parse_int,
                "core_grid_x": parse_int,
                "core_grid_y": parse_int,
                "enqueue_finish_us": parse_int,
                "max_abs_error": parse_float,
            }
            for field, parser in numeric_fields.items():
                if field in record:
                    record[field] = parser(record[field])
            if "ok" in record:
                record["ok"] = record["ok"] == "1"

            results.append(record)
    return results


def summarize_host(results, drop_first_repeat=False):
    grouped = defaultdict(list)
    for result in results:
        key = (
            result["op"],
            result["mode"],
            result["tiles"],
            result["num_slots"],
            result["slot_bytes"],
            result["chain_depth"],
            result.get("matmul_m_tiles", 0),
            result.get("matmul_n_tiles", 0),
            result["matmul_k_tiles"],
            result.get("core_grid_x", 1),
            result.get("core_grid_y", 1),
        )
        grouped[key].append(result)

    rows = []
    for key in sorted(grouped):
        samples = grouped[key]
        if drop_first_repeat and len(samples) > 1:
            first_repeat = min(sample["repeat"] for sample in samples)
            samples = [sample for sample in samples if sample["repeat"] != first_repeat]
        enqueue_values = [sample["enqueue_finish_us"] for sample in samples]
        max_errors = [sample["max_abs_error"] for sample in samples]
        output_tile_count = samples[0].get("output_tiles", samples[0]["tiles"])
        input_pair_count = samples[0].get("input_tile_pairs", output_tile_count)
        rows.append(
            {
                "op": key[0],
                "mode": key[1],
                "tiles": key[2],
                "num_slots": key[3],
                "slot_bytes": key[4],
                "chain_depth": key[5],
                "matmul_m_tiles": key[6],
                "matmul_n_tiles": key[7],
                "matmul_k_tiles": key[8],
                "core_grid_x": key[9],
                "core_grid_y": key[10],
                "count": len(samples),
                "ok_all": all(sample["ok"] for sample in samples),
                "median_enqueue_finish_us": median(enqueue_values),
                "min_enqueue_finish_us": min(enqueue_values),
                "max_enqueue_finish_us": max(enqueue_values),
                "max_abs_error": max(max_errors),
                "median_us_per_output_tile": median(enqueue_values) / max(1, output_tile_count),
                "median_us_per_input_pair": median(enqueue_values) / max(1, input_pair_count),
            }
        )
    return rows


def empty_zone_metadata():
    return {
        "op": "",
        "mode": "",
        "stage": "",
        "tiles": 0,
        "num_slots": 0,
        "chain_depth": 0,
        "matmul_m_tiles": 0,
        "matmul_n_tiles": 0,
        "matmul_k_tiles": 0,
        "core_grid_x": 0,
        "core_grid_y": 0,
        "local_output_tiles": 0,
        "local_input_tile_pairs": 0,
    }


def parse_zone_name(zone):
    metadata = empty_zone_metadata()

    match = TILE_ZONE_RE.match(zone)
    if match:
        fields = match.groupdict()
        tiles = int(fields["tiles"])
        metadata.update(
            {
                "op": OP_LABELS[fields["op"]],
                "mode": MODE_LABELS[fields["mode"]],
                "stage": STAGE_LABELS[fields["stage"]],
                "tiles": tiles,
                "num_slots": int(fields["num_slots"]),
                "chain_depth": int(fields["chain_depth"]),
                "core_grid_x": 1,
                "core_grid_y": 1,
                "local_output_tiles": tiles,
                "local_input_tile_pairs": tiles,
            }
        )
        return metadata

    match = MATMUL_ZONE_RE.match(zone)
    if match:
        fields = match.groupdict()
        matmul_m_tiles = int(fields["matmul_m_tiles"])
        matmul_n_tiles = int(fields["matmul_n_tiles"])
        matmul_k_tiles = int(fields["matmul_k_tiles"])
        core_grid_x = int(fields["core_grid_x"])
        core_grid_y = int(fields["core_grid_y"])
        local_m_tiles = matmul_m_tiles // max(1, core_grid_y)
        local_n_tiles = matmul_n_tiles // max(1, core_grid_x)
        local_output_tiles = local_m_tiles * local_n_tiles
        metadata.update(
            {
                "op": OP_LABELS[fields["op"]],
                "mode": MODE_LABELS[fields["mode"]],
                "stage": STAGE_LABELS[fields["stage"]],
                "tiles": matmul_m_tiles * matmul_n_tiles,
                "num_slots": int(fields["num_slots"]),
                "chain_depth": 1,
                "matmul_m_tiles": matmul_m_tiles,
                "matmul_n_tiles": matmul_n_tiles,
                "matmul_k_tiles": matmul_k_tiles,
                "core_grid_x": core_grid_x,
                "core_grid_y": core_grid_y,
                "local_output_tiles": local_output_tiles,
                "local_input_tile_pairs": local_output_tiles * matmul_k_tiles,
            }
        )
        return metadata

    match = OLD_ZONE_RE.match(zone)
    if match:
        fields = match.groupdict()
        metadata.update(
            {
                "op": OP_LABELS[fields["op"]],
                "mode": MODE_LABELS[fields["mode"]],
                "stage": STAGE_LABELS[fields["stage"]],
            }
        )
    return metadata


def local_counts_from_case(case):
    if case["matmul_m_tiles"] and case["matmul_n_tiles"]:
        local_m_tiles = case["matmul_m_tiles"] // max(1, case["core_grid_y"])
        local_n_tiles = case["matmul_n_tiles"] // max(1, case["core_grid_x"])
        local_output_tiles = local_m_tiles * local_n_tiles
        return local_output_tiles, local_output_tiles * max(1, case["matmul_k_tiles"])
    return case["tiles"], case["tiles"]


def single_host_case_metadata(host_summary):
    if not host_summary:
        return None
    keys = {
        (
            row["op"],
            row["tiles"],
            row["num_slots"],
            row["chain_depth"],
            row["matmul_m_tiles"],
            row["matmul_n_tiles"],
            row["matmul_k_tiles"],
            row["core_grid_x"],
            row["core_grid_y"],
        )
        for row in host_summary
    }
    if len(keys) != 1:
        return None

    row = host_summary[0]
    local_output_tiles, local_input_tile_pairs = local_counts_from_case(row)
    return {
        "tiles": row["tiles"],
        "num_slots": row["num_slots"],
        "chain_depth": row["chain_depth"],
        "matmul_m_tiles": row["matmul_m_tiles"],
        "matmul_n_tiles": row["matmul_n_tiles"],
        "matmul_k_tiles": row["matmul_k_tiles"],
        "core_grid_x": row["core_grid_x"],
        "core_grid_y": row["core_grid_y"],
        "local_output_tiles": local_output_tiles,
        "local_input_tile_pairs": local_input_tile_pairs,
    }


def enrich_single_case_zone_metadata(zone_summary, host_summary):
    case = single_host_case_metadata(host_summary)
    if case is None:
        return zone_summary
    for row in zone_summary:
        if row["op"] and row["tiles"] == 0:
            row.update(case)
    return zone_summary


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
        if not zone.startswith("SPM_"):
            continue

        key = (zone, row[core_x_col].strip(), row[core_y_col].strip(), row[risc_col].strip())
        timestamp = int(row[time_col].strip())
        phase = row[phase_col].strip()
        if phase_is_start(phase):
            open_zones[key].append(timestamp)
        elif phase_is_end(phase) and open_zones[key]:
            start = open_zones[key].pop()
            durations[key].append(timestamp - start)

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


def write_csv(path, rows):
    if path is None or not rows:
        return
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(title, rows, fields):
    if not rows:
        return
    print(title)
    print(",".join(fields))
    for row in rows:
        print(",".join(str(row[field]) for field in fields))
    print()


def build_mode_comparisons(host_summary):
    grouped = defaultdict(dict)
    for row in host_summary:
        key = (
            row["op"],
            row["tiles"],
            row["num_slots"],
            row["slot_bytes"],
            row["chain_depth"],
            row["matmul_m_tiles"],
            row["matmul_n_tiles"],
            row["matmul_k_tiles"],
            row["core_grid_x"],
            row["core_grid_y"],
        )
        grouped[key][row["mode"]] = row

    comparisons = []
    for key, modes in grouped.items():
        if "cb" in modes and "static-runtime" in modes:
            cb = modes["cb"]
            static_runtime = modes["static-runtime"]
            comparisons.append(
                {
                    "op": key[0],
                    "tiles": key[1],
                    "num_slots": key[2],
                    "slot_bytes": key[3],
                    "chain_depth": key[4],
                    "matmul_m_tiles": key[5],
                    "matmul_n_tiles": key[6],
                    "matmul_k_tiles": key[7],
                    "core_grid_x": key[8],
                    "core_grid_y": key[9],
                    "lhs_mode": "cb",
                    "rhs_mode": "static-runtime",
                    "delta_enqueue_finish_us": cb["median_enqueue_finish_us"] - static_runtime["median_enqueue_finish_us"],
                    "delta_us_per_output_tile": cb["median_us_per_output_tile"] - static_runtime["median_us_per_output_tile"],
                    "delta_us_per_input_pair": cb["median_us_per_input_pair"] - static_runtime["median_us_per_input_pair"],
                }
            )
        if "static-runtime" in modes and "static-compiletime" in modes:
            runtime = modes["static-runtime"]
            compiletime = modes["static-compiletime"]
            comparisons.append(
                {
                    "op": key[0],
                    "tiles": key[1],
                    "num_slots": key[2],
                    "slot_bytes": key[3],
                    "chain_depth": key[4],
                    "matmul_m_tiles": key[5],
                    "matmul_n_tiles": key[6],
                    "matmul_k_tiles": key[7],
                    "core_grid_x": key[8],
                    "core_grid_y": key[9],
                    "lhs_mode": "static-runtime",
                    "rhs_mode": "static-compiletime",
                    "delta_enqueue_finish_us": runtime["median_enqueue_finish_us"] - compiletime["median_enqueue_finish_us"],
                    "delta_us_per_output_tile": runtime["median_us_per_output_tile"] - compiletime["median_us_per_output_tile"],
                    "delta_us_per_input_pair": runtime["median_us_per_input_pair"] - compiletime["median_us_per_input_pair"],
                }
            )
        if "static-serialized" in modes and "static-runtime" in modes:
            serialized = modes["static-serialized"]
            runtime = modes["static-runtime"]
            comparisons.append(
                {
                    "op": key[0],
                    "tiles": key[1],
                    "num_slots": key[2],
                    "slot_bytes": key[3],
                    "chain_depth": key[4],
                    "matmul_m_tiles": key[5],
                    "matmul_n_tiles": key[6],
                    "matmul_k_tiles": key[7],
                    "core_grid_x": key[8],
                    "core_grid_y": key[9],
                    "lhs_mode": "static-serialized",
                    "rhs_mode": "static-runtime",
                    "delta_enqueue_finish_us": serialized["median_enqueue_finish_us"] - runtime["median_enqueue_finish_us"],
                    "delta_us_per_output_tile": serialized["median_us_per_output_tile"] - runtime["median_us_per_output_tile"],
                    "delta_us_per_input_pair": serialized["median_us_per_input_pair"] - runtime["median_us_per_input_pair"],
                }
            )
        for streamreg_mode in ("static-streamreg-cbregs", "static-streamreg-cbregs-compiletime"):
            if "cb" in modes and streamreg_mode in modes:
                cb = modes["cb"]
                streamreg = modes[streamreg_mode]
                comparisons.append(
                    {
                        "op": key[0],
                        "tiles": key[1],
                        "num_slots": key[2],
                        "slot_bytes": key[3],
                        "chain_depth": key[4],
                        "matmul_m_tiles": key[5],
                        "matmul_n_tiles": key[6],
                        "matmul_k_tiles": key[7],
                        "core_grid_x": key[8],
                        "core_grid_y": key[9],
                        "lhs_mode": "cb",
                        "rhs_mode": streamreg_mode,
                        "delta_enqueue_finish_us": cb["median_enqueue_finish_us"] - streamreg["median_enqueue_finish_us"],
                        "delta_us_per_output_tile": cb["median_us_per_output_tile"] - streamreg["median_us_per_output_tile"],
                        "delta_us_per_input_pair": cb["median_us_per_input_pair"] - streamreg["median_us_per_input_pair"],
                    }
                )
            if streamreg_mode in modes and "static-runtime" in modes:
                streamreg = modes[streamreg_mode]
                runtime = modes["static-runtime"]
                comparisons.append(
                    {
                        "op": key[0],
                        "tiles": key[1],
                        "num_slots": key[2],
                        "slot_bytes": key[3],
                        "chain_depth": key[4],
                        "matmul_m_tiles": key[5],
                        "matmul_n_tiles": key[6],
                        "matmul_k_tiles": key[7],
                        "core_grid_x": key[8],
                        "core_grid_y": key[9],
                        "lhs_mode": streamreg_mode,
                        "rhs_mode": "static-runtime",
                        "delta_enqueue_finish_us": streamreg["median_enqueue_finish_us"] - runtime["median_enqueue_finish_us"],
                        "delta_us_per_output_tile": streamreg["median_us_per_output_tile"] - runtime["median_us_per_output_tile"],
                        "delta_us_per_input_pair": streamreg["median_us_per_input_pair"] - runtime["median_us_per_input_pair"],
                    }
                )
    return comparisons


def build_device_mode_comparisons(zone_summary):
    grouped = defaultdict(dict)
    for row in zone_summary:
        if not row["op"] or not row["mode"]:
            continue
        key = (
            row["op"],
            row["tiles"],
            row["num_slots"],
            row["chain_depth"],
            row["matmul_m_tiles"],
            row["matmul_n_tiles"],
            row["matmul_k_tiles"],
            row["core_grid_x"],
            row["core_grid_y"],
            row["stage"],
            row["core"],
            row["risc"],
        )
        grouped[key][row["mode"]] = row

    comparisons = []
    comparison_pairs = [
        ("cb", "static-runtime"),
        ("cb", "static-streamreg-cbregs"),
        ("cb", "static-streamreg-cbregs-compiletime"),
        ("static-runtime", "static-compiletime"),
        ("static-serialized", "static-runtime"),
        ("static-streamreg-cbregs", "static-runtime"),
        ("static-runtime", "static-streamreg-cbregs-compiletime"),
        ("static-streamreg-cbregs", "static-streamreg-cbregs-compiletime"),
    ]
    for key, modes in sorted(grouped.items()):
        for lhs_mode, rhs_mode in comparison_pairs:
            if lhs_mode not in modes or rhs_mode not in modes:
                continue
            lhs = modes[lhs_mode]
            rhs = modes[rhs_mode]
            local_output_tiles = max(1, lhs["local_output_tiles"])
            local_input_tile_pairs = max(1, lhs["local_input_tile_pairs"])
            delta_cycles = lhs["median_cycles"] - rhs["median_cycles"]
            comparisons.append(
                {
                    "op": key[0],
                    "tiles": key[1],
                    "num_slots": key[2],
                    "chain_depth": key[3],
                    "matmul_m_tiles": key[4],
                    "matmul_n_tiles": key[5],
                    "matmul_k_tiles": key[6],
                    "core_grid_x": key[7],
                    "core_grid_y": key[8],
                    "stage": key[9],
                    "core": key[10],
                    "risc": key[11],
                    "lhs_mode": lhs_mode,
                    "rhs_mode": rhs_mode,
                    "delta_median_cycles": delta_cycles,
                    "delta_cycles_per_local_output_tile": delta_cycles / local_output_tiles,
                    "delta_cycles_per_local_input_pair": delta_cycles / local_input_tile_pairs,
                }
            )
    return comparisons


def main():
    args = parse_args()

    host_results = parse_host_results(args.host_log)
    host_summary = summarize_host(host_results, args.drop_first_repeat)
    zone_summary = enrich_single_case_zone_metadata(parse_device_zones(args.device_csv), host_summary)
    comparisons = build_mode_comparisons(host_summary)
    device_comparisons = build_device_mode_comparisons(zone_summary)

    write_csv(args.summary_csv, host_summary)
    write_csv(args.zones_csv, zone_summary)
    write_csv(args.device_comparison_csv, device_comparisons)

    print_table(
        "host_summary",
        host_summary,
        [
            "op",
            "mode",
            "tiles",
            "num_slots",
            "chain_depth",
            "matmul_m_tiles",
            "matmul_n_tiles",
            "matmul_k_tiles",
            "core_grid_x",
            "core_grid_y",
            "count",
            "ok_all",
            "median_enqueue_finish_us",
            "median_us_per_output_tile",
            "median_us_per_input_pair",
            "max_abs_error",
        ],
    )
    print_table(
        "device_zone_summary",
        zone_summary,
        [
            "zone",
            "op",
            "mode",
            "stage",
            "tiles",
            "num_slots",
            "chain_depth",
            "matmul_m_tiles",
            "matmul_n_tiles",
            "matmul_k_tiles",
            "core_grid_x",
            "core_grid_y",
            "local_output_tiles",
            "local_input_tile_pairs",
            "core",
            "risc",
            "count",
            "median_cycles",
            "min_cycles",
            "max_cycles",
        ],
    )
    print_table(
        "mode_comparison",
        comparisons,
        [
            "op",
            "tiles",
            "num_slots",
            "slot_bytes",
            "chain_depth",
            "matmul_m_tiles",
            "matmul_n_tiles",
            "matmul_k_tiles",
            "core_grid_x",
            "core_grid_y",
            "lhs_mode",
            "rhs_mode",
            "delta_enqueue_finish_us",
            "delta_us_per_output_tile",
            "delta_us_per_input_pair",
        ],
    )
    print_table(
        "device_mode_comparison",
        device_comparisons,
        [
            "op",
            "tiles",
            "num_slots",
            "chain_depth",
            "matmul_m_tiles",
            "matmul_n_tiles",
            "matmul_k_tiles",
            "core_grid_x",
            "core_grid_y",
            "stage",
            "core",
            "risc",
            "lhs_mode",
            "rhs_mode",
            "delta_median_cycles",
            "delta_cycles_per_local_output_tile",
            "delta_cycles_per_local_input_pair",
        ],
    )


if __name__ == "__main__":
    main()
