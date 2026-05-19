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


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize CB protocol profiler zones.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--prefix", default="CBP_")
    return parser.parse_args()


def find_header(rows):
    for index, row in enumerate(rows):
        normalized = [cell.strip() for cell in row]
        if "zone name" in normalized and "time[cycles since reset]" in normalized:
            return index, normalized
    raise RuntimeError("Could not find profiler CSV header")


def phase_is_start(value):
    value = value.strip().lower()
    return value in {"zone_start", "begin", "start"}


def phase_is_end(value):
    value = value.strip().lower()
    return value in {"zone_end", "end", "stop"}


def main():
    args = parse_args()
    if args.iterations <= 0:
        raise RuntimeError("--iterations must be greater than zero")

    with args.csv_path.open(newline="") as csv_file:
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
        if not zone.startswith(args.prefix):
            continue

        key = (row[core_x_col].strip(), row[core_y_col].strip(), row[risc_col].strip(), zone)
        timestamp = int(row[time_col].strip())
        phase = row[phase_col].strip()

        if phase_is_start(phase):
            open_zones[key].append(timestamp)
        elif phase_is_end(phase):
            if not open_zones[key]:
                continue
            start = open_zones[key].pop()
            durations[key].append(timestamp - start)

    zone_medians = {}

    print("zone,core,risc,count,median_cycles,min_cycles,max_cycles,median_cycles_per_iter")
    for key in sorted(durations):
        core_x, core_y, risc, zone = key
        samples = durations[key]
        median_cycles = statistics.median(samples)
        zone_medians[zone] = median_cycles
        print(
            f"{zone},({core_x},{core_y}),{risc},{len(samples)},"
            f"{median_cycles:.0f},{min(samples)},{max(samples)},"
            f"{median_cycles / args.iterations:.4f}"
        )

    local_modes = (
        ("empty", "CBP_EMPTY"),
        ("cb-get-write-ptr", "CBP_GET_WRITE_PTR"),
        ("cb-get-read-write-ptr", "CBP_GET_READ_WRITE_PTR"),
        ("cb-get-tile-size", "CBP_GET_TILE_SIZE"),
        ("cb-api-roundtrip", "CBP_API_ROUNDTRIP"),
        ("static-ring", "CBP_STATIC_RING"),
        ("static-counter", "CBP_STATIC_COUNTER"),
        ("system-static-nosync", "CBP_SYSTEM_STATIC_NOSYNC"),
        ("dram-single-nosync", "CBP_DRAM_SINGLE_NOSYNC"),
    )
    empty_cycles = zone_medians.get("CBP_EMPTY")
    if empty_cycles is not None:
        print()
        print("local_scenario,cycles,cycles_per_iter,net_vs_empty_per_iter")
        for scenario, zone in local_modes:
            if zone not in zone_medians:
                continue
            cycles = zone_medians[zone]
            print(
                f"{scenario},{cycles:.0f},{cycles / args.iterations:.4f},"
                f"{(cycles - empty_cycles) / args.iterations:.4f}"
            )

    cross = defaultdict(dict)
    cross_pattern = re.compile(r"^CBP_CROSS_(EMPTY|CB|STATIC|STREAMREG)_(PRODUCER|CONSUMER)$")
    for zone, median_cycles in zone_medians.items():
        match = cross_pattern.match(zone)
        if match:
            scenario, role = match.groups()
            cross[scenario.lower()][role.lower()] = median_cycles

    if cross:
        print()
        print("cross_scenario,producer_cycles,consumer_cycles,pair_max_cycles,pair_max_cycles_per_iter,net_vs_empty_per_iter")
        empty_pair_cycles = None
        if "empty" in cross and {"producer", "consumer"} <= cross["empty"].keys():
            empty_pair_cycles = max(cross["empty"]["producer"], cross["empty"]["consumer"])

        for scenario in ("empty", "cb", "static", "streamreg"):
            roles = cross.get(scenario)
            if not roles or not {"producer", "consumer"} <= roles.keys():
                continue

            pair_cycles = max(roles["producer"], roles["consumer"])
            net_cycles_per_iter = ""
            if empty_pair_cycles is not None:
                net_cycles_per_iter = f"{(pair_cycles - empty_pair_cycles) / args.iterations:.4f}"
            print(
                f"cross-{scenario},{roles['producer']:.0f},{roles['consumer']:.0f},"
                f"{pair_cycles:.0f},{pair_cycles / args.iterations:.4f},{net_cycles_per_iter}"
            )

        comparisons = (
            ("cross-cb-minus-cross-static", "cb", "static"),
            ("cross-cb-minus-cross-streamreg", "cb", "streamreg"),
            ("cross-streamreg-minus-cross-static", "streamreg", "static"),
        )
        printed_header = False
        for label, lhs, rhs in comparisons:
            lhs_roles = cross.get(lhs)
            rhs_roles = cross.get(rhs)
            if not lhs_roles or not rhs_roles:
                continue
            if not ({"producer", "consumer"} <= lhs_roles.keys() and {"producer", "consumer"} <= rhs_roles.keys()):
                continue
            lhs_pair_cycles = max(lhs_roles["producer"], lhs_roles["consumer"])
            rhs_pair_cycles = max(rhs_roles["producer"], rhs_roles["consumer"])
            if not printed_header:
                print()
                print("comparison,delta_cycles,delta_cycles_per_iter")
                printed_header = True
            delta = lhs_pair_cycles - rhs_pair_cycles
            print(f"{label},{delta:.0f},{delta / args.iterations:.4f}")

    system = defaultdict(dict)
    system_pattern = re.compile(r"^CBP_SYSTEM_(CB|STATIC_RUNTIME|STATIC_COMPILETIME|STREAMREG)_(PRODUCER|CONSUMER)$")
    for zone, median_cycles in zone_medians.items():
        match = system_pattern.match(zone)
        if match:
            scenario, role = match.groups()
            system[scenario.lower().replace("_", "-")][role.lower()] = median_cycles

    if system:
        print()
        print(
            "system_scenario,producer_cycles,consumer_cycles,pair_max_cycles,"
            "pair_max_cycles_per_iter,net_vs_cross_empty_per_iter"
        )
        empty_pair_cycles = None
        if "empty" in cross and {"producer", "consumer"} <= cross["empty"].keys():
            empty_pair_cycles = max(cross["empty"]["producer"], cross["empty"]["consumer"])

        for scenario in ("cb", "static-runtime", "static-compiletime", "streamreg"):
            roles = system.get(scenario)
            if not roles or not {"producer", "consumer"} <= roles.keys():
                continue

            pair_cycles = max(roles["producer"], roles["consumer"])
            net_cycles_per_iter = ""
            if empty_pair_cycles is not None:
                net_cycles_per_iter = f"{(pair_cycles - empty_pair_cycles) / args.iterations:.4f}"
            print(
                f"system-{scenario},{roles['producer']:.0f},{roles['consumer']:.0f},"
                f"{pair_cycles:.0f},{pair_cycles / args.iterations:.4f},{net_cycles_per_iter}"
            )

        comparisons = (
            ("system-cb-minus-static-runtime", "cb", "static-runtime"),
            ("system-static-runtime-minus-compiletime", "static-runtime", "static-compiletime"),
            ("system-cb-minus-static-compiletime", "cb", "static-compiletime"),
            ("system-cb-minus-streamreg", "cb", "streamreg"),
            ("system-streamreg-minus-static-runtime", "streamreg", "static-runtime"),
        )
        printed_header = False
        for label, lhs, rhs in comparisons:
            lhs_roles = system.get(lhs)
            rhs_roles = system.get(rhs)
            if not lhs_roles or not rhs_roles:
                continue
            if not ({"producer", "consumer"} <= lhs_roles.keys() and {"producer", "consumer"} <= rhs_roles.keys()):
                continue
            lhs_pair_cycles = max(lhs_roles["producer"], lhs_roles["consumer"])
            rhs_pair_cycles = max(rhs_roles["producer"], rhs_roles["consumer"])
            if not printed_header:
                print()
                print("system_comparison,delta_cycles,delta_cycles_per_iter")
                printed_header = True
            delta = lhs_pair_cycles - rhs_pair_cycles
            print(f"{label},{delta:.0f},{delta / args.iterations:.4f}")

    dram = defaultdict(dict)
    dram_pattern = re.compile(r"^CBP_DRAM_(CB|STATIC_RUNTIME|STATIC_COMPILETIME|STREAMREG)_(PRODUCER|CONSUMER)$")
    for zone, median_cycles in zone_medians.items():
        match = dram_pattern.match(zone)
        if match:
            scenario, role = match.groups()
            dram[scenario.lower().replace("_", "-")][role.lower()] = median_cycles

    if dram:
        print()
        print(
            "dram_scenario,producer_cycles,consumer_cycles,pair_max_cycles,"
            "pair_max_cycles_per_iter,net_vs_cross_empty_per_iter"
        )
        empty_pair_cycles = None
        if "empty" in cross and {"producer", "consumer"} <= cross["empty"].keys():
            empty_pair_cycles = max(cross["empty"]["producer"], cross["empty"]["consumer"])

        for scenario in ("cb", "static-runtime", "static-compiletime", "streamreg"):
            roles = dram.get(scenario)
            if not roles or not {"producer", "consumer"} <= roles.keys():
                continue

            pair_cycles = max(roles["producer"], roles["consumer"])
            net_cycles_per_iter = ""
            if empty_pair_cycles is not None:
                net_cycles_per_iter = f"{(pair_cycles - empty_pair_cycles) / args.iterations:.4f}"
            print(
                f"dram-{scenario},{roles['producer']:.0f},{roles['consumer']:.0f},"
                f"{pair_cycles:.0f},{pair_cycles / args.iterations:.4f},{net_cycles_per_iter}"
            )

        comparisons = (
            ("dram-cb-minus-static-runtime", "cb", "static-runtime"),
            ("dram-static-runtime-minus-compiletime", "static-runtime", "static-compiletime"),
            ("dram-cb-minus-static-compiletime", "cb", "static-compiletime"),
            ("dram-cb-minus-streamreg", "cb", "streamreg"),
            ("dram-streamreg-minus-static-runtime", "streamreg", "static-runtime"),
        )
        printed_header = False
        for label, lhs, rhs in comparisons:
            lhs_roles = dram.get(lhs)
            rhs_roles = dram.get(rhs)
            if not lhs_roles or not rhs_roles:
                continue
            if not ({"producer", "consumer"} <= lhs_roles.keys() and {"producer", "consumer"} <= rhs_roles.keys()):
                continue
            lhs_pair_cycles = max(lhs_roles["producer"], lhs_roles["consumer"])
            rhs_pair_cycles = max(rhs_roles["producer"], rhs_roles["consumer"])
            if not printed_header:
                print()
                print("dram_comparison,delta_cycles,delta_cycles_per_iter")
                printed_header = True
            delta = lhs_pair_cycles - rhs_pair_cycles
            print(f"{label},{delta:.0f},{delta / args.iterations:.4f}")


if __name__ == "__main__":
    main()
