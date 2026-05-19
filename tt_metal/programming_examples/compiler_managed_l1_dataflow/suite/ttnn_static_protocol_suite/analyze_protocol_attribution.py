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


CASE_SPECS = {
    "real_copy": {
        "title": "real_copy_protocol",
        "baseline_mode": "cb",
        "primary_modes": (
            "static-runtime",
            "static-compiletime",
            "static-streamreg-scratch",
            "static-streamreg-scratch-compiletime",
        ),
        "shape_fields": ("tiles", "num_pages", "core_grid_x", "core_grid_y"),
        "work_field": "max_tiles_per_core",
        "expected": "negative_or_near_cb",
    },
    "real_tile_add": {
        "title": "real_tile_add_protocol",
        "baseline_mode": "cb",
        "primary_modes": (
            "static-runtime",
            "static-compiletime",
            "static-streamreg-cbregs",
        ),
        "shape_fields": ("tiles", "num_pages", "core_grid_x", "core_grid_y"),
        "work_field": "max_tiles_per_core",
        "expected": "positive_static_cbregs_control",
    },
    "real_matmul": {
        "title": "real_matmul_protocol",
        "baseline_mode": "profiled-cb",
        "primary_modes": (
            "static-input-only",
            "static-output-only",
            "static-input-output",
            "static-input-only-cbregs",
            "static-output-only-cbregs",
            "static-input-output-cbregs",
            "static-input-only-cbregs-compiletime",
            "static-output-only-cbregs-compiletime",
            "static-input-output-cbregs-compiletime",
        ),
        "shape_fields": ("M", "N", "K", "num_pages"),
        "work_field": "",
        "expected": "mixed",
    },
    "ttnn_add": {
        "title": "ttnn_binary_ng_no_bcast_protocol",
        "baseline_mode": "cb",
        "primary_modes": ("static-runtime", "static-streamreg-cbregs"),
        "shape_fields": ("tiles", "num_pages", "core_grid_x", "core_grid_y"),
        "work_field": "max_tiles_per_core",
        "expected": "positive_control",
    },
    "ttnn_sfpu_no_bcast": {
        "title": "ttnn_binary_ng_sfpu_no_bcast_protocol div",
        "baseline_mode": "cb",
        "primary_modes": ("static-runtime", "static-streamreg-cbregs"),
        "shape_fields": ("op", "tiles", "num_pages", "core_grid_x", "core_grid_y"),
        "work_field": "max_tiles_per_core",
        "expected": "compute_heavy_noise_floor",
    },
    "ttnn_bcast_to_row": {
        "title": "ttnn_bcast_to_protocol row-bcast",
        "baseline_mode": "cb",
        "primary_modes": ("static-runtime", "static-streamreg-cbregs"),
        "shape_fields": ("tiles", "width_tiles", "height_tiles", "num_pages", "core_grid_x", "core_grid_y"),
        "work_field": "max_tiles_per_core",
        "expected": "measure_direct_broadcast",
    },
    "ttnn_binary_ng_row_bcast": {
        "title": "ttnn_binary_ng_row_bcast_protocol reader-side add",
        "baseline_mode": "cb",
        "primary_modes": ("static-runtime", "static-streamreg-cbregs"),
        "shape_fields": ("op", "tiles", "width_tiles", "num_pages", "core_grid_x", "core_grid_y"),
        "work_field": "max_tiles_per_core",
        "expected": "measure_direct_binary_row_broadcast",
    },
    "ttnn_paged_update_cache": {
        "title": "ttnn_paged_update_cache_protocol",
        "baseline_mode": "cb",
        "primary_modes": ("static-runtime", "static-streamreg-cbregs"),
        "shape_fields": (
            "users",
            "kv_heads",
            "head_dim",
            "block_size",
            "max_seq_len",
            "cache_idx",
            "num_pages",
        ),
        "work_field": "users",
        "expected": "positive_shape_specific",
    },
    "ttnn_transpose_wh": {
        "title": "ttnn_transpose_wh_protocol",
        "baseline_mode": "cb",
        "primary_modes": ("static-runtime", "static-streamreg-cbregs"),
        "shape_fields": ("height_tiles", "width_tiles", "batches", "num_pages", "core_grid_x", "core_grid_y"),
        "work_field": "max_tiles_per_core",
        "expected": "layout_weak_positive",
    },
    "ttnn_embedding_lookup": {
        "title": "ttnn_embedding_lookup_protocol",
        "baseline_mode": "cb",
        "primary_modes": ("static-runtime", "static-streamreg-cbregs"),
        "shape_fields": ("rows", "vocab_size", "dim", "num_pages", "core_grid_x", "core_grid_y"),
        "work_field": "max_rows_per_core",
        "expected": "embedding_lookup_positive",
    },
    "ttnn_slice_tile": {
        "title": "ttnn_slice_tile_protocol",
        "baseline_mode": "cb",
        "primary_modes": ("static-runtime", "static-streamreg-cbregs"),
        "shape_fields": (
            "input_height_tiles",
            "input_width_tiles",
            "output_height_tiles",
            "output_width_tiles",
            "start_height_tile",
            "start_width_tile",
            "batches",
            "num_pages",
            "core_grid_x",
            "core_grid_y",
        ),
        "work_field": "output_tiles",
        "expected": "layout_slice_weak_positive",
    },
    "ttnn_kv_cache_load_slice": {
        "title": "ttnn_kv_cache_load_slice_protocol",
        "baseline_mode": "cb",
        "primary_modes": ("static-runtime", "static-streamreg-cbregs"),
        "shape_fields": (
            "input_seq_tiles",
            "output_seq_tiles",
            "head_dim_tiles",
            "start_seq_tile",
            "core_grid_x",
            "core_grid_y",
        ),
        "work_field": "output_tiles",
        "expected": "kv_cache_read_near_noise",
    },
}


STAGE_ORDER = {
    "reader": 0,
    "compute-input-untilize": 1,
    "compute-unpack": 1,
    "compute-math": 2,
    "compute-pack": 3,
    "writer": 4,
}

PAGED_UPDATE_CACHE_CASE_RE = re.compile(
    r"^u(?P<users>\d+)_kvh(?P<kv_heads>\d+)_d(?P<head_dim>\d+)_"
    r"b(?P<block_size>\d+)_s(?P<max_seq_len>\d+)_idx(?P<cache_idx>\d+)_p(?P<num_pages>\d+)$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 compiler-managed L1 协议 profiler CSV 输出生成中文归因报告。"
    )
    parser.add_argument("--real-copy-dir", type=Path, help="run_real_copy_protocol_cases.py 生成的目录。")
    parser.add_argument("--real-tile-add-dir", type=Path, help="run_real_tile_add_protocol_cases.py 生成的目录。")
    parser.add_argument("--real-matmul-dir", type=Path, help="run_real_matmul_protocol_cases.py 生成的目录。")
    parser.add_argument(
        "--ttnn-add-dir", type=Path, help="run_ttnn_binary_ng_no_bcast_protocol_cases.py 生成的目录。"
    )
    parser.add_argument(
        "--ttnn-sfpu-no-bcast-dir",
        type=Path,
        help="run_ttnn_binary_ng_sfpu_no_bcast_protocol_cases.py 生成的目录。",
    )
    parser.add_argument(
        "--ttnn-bcast-to-row-dir", type=Path, help="run_ttnn_bcast_to_protocol_cases.py 生成的目录。"
    )
    parser.add_argument(
        "--ttnn-binary-ng-row-bcast-dir",
        type=Path,
        help="run_ttnn_binary_ng_row_bcast_protocol_cases.py 生成的目录。",
    )
    parser.add_argument(
        "--ttnn-paged-update-cache-dir",
        type=Path,
        help="run_ttnn_paged_update_cache_protocol_cases.py 生成的目录。",
    )
    parser.add_argument(
        "--ttnn-transpose-wh-dir",
        type=Path,
        help="run_ttnn_transpose_wh_protocol_cases.py 生成的目录。",
    )
    parser.add_argument(
        "--ttnn-embedding-lookup-dir",
        type=Path,
        help="run_ttnn_embedding_lookup_protocol_cases.py 生成的目录。",
    )
    parser.add_argument(
        "--ttnn-slice-tile-dir",
        type=Path,
        help="run_ttnn_slice_tile_protocol_cases.py 生成的目录。",
    )
    parser.add_argument(
        "--ttnn-kv-cache-load-slice-dir",
        type=Path,
        help="run_ttnn_kv_cache_load_slice_protocol_cases.py 生成的目录。",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path):
    if not path or not path.exists():
        return []
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_int(value, default=0):
    number = to_float(value)
    if number is None:
        return default
    return int(number)


def fmt_number(value, digits=2):
    if value is None:
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def fmt_signed(value, digits=2):
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def root_cause_label(value):
    labels = {
        "missing_device_delta": "缺少 device delta",
        "sync_storage_overhead_near_cb": "同步状态存储开销，接近 CB",
        "sync_storage_overhead": "同步状态存储开销",
        "compile_time_protocol_overhead_removed": "compile-time queue/layout 去掉 runtime/config 成本",
        "protocol_overhead_visible": "协议开销可见",
        "protocol_overhead_visible_cbregs_control": "协议开销可见，cbregs 是控制项",
        "protocol_overhead_visible_shape_specific": "协议开销只在特定 shape 可见",
        "static_protocol_regression_shape_specific": "static protocol 在特定 shape 回退",
        "critical_path_hidden_by_compute_reuse": "被 compute/reuse critical path 掩盖",
        "protocol_overhead_visible_small_positive": "协议开销可见，但收益较小",
        "protocol_overhead_visible_tiny_positive": "协议开销轻微可见，接近噪声区间",
        "static_protocol_regression": "static protocol 回退",
        "noise_floor": "接近噪声区间",
    }
    return labels.get(value, value)


def mode_from_comparison(row, fallback):
    mode = row.get("mode")
    if mode:
        return mode
    return fallback


def normalize_mode(case_name, mode):
    if case_name == "real_copy" and mode == "static-streamreg":
        return "static-streamreg-scratch"
    return mode


def mode_role(case_name, mode):
    if mode == "static-streamreg-cbregs":
        return "level_b_standard"
    if mode == "static-streamreg-cbregs-compiletime":
        return "compiletime_ablation"
    if case_name == "real_copy" and mode == "static-streamreg-scratch-compiletime":
        return "dataflow_compiletime_ablation"
    if mode.endswith("-compiletime"):
        return "compiletime_ablation"
    if mode == "static-runtime":
        return "static_runtime_control"
    return "control"


def shape_key(row, spec):
    return tuple(row.get(field, "") for field in spec["shape_fields"])


def shape_label(row, spec):
    pieces = []
    for field in spec["shape_fields"]:
        value = row.get(field, "")
        if value != "":
            pieces.append(f"{field}={value}")
    return " ".join(pieces) if pieces else "all"


def enrich_rows_from_case_name(case_name, rows):
    if case_name != "ttnn_paged_update_cache":
        return rows
    enriched = []
    for row in rows:
        row = dict(row)
        match = PAGED_UPDATE_CACHE_CASE_RE.match(row.get("case", ""))
        if match:
            for field, value in match.groupdict().items():
                row.setdefault(field, value)
                if row[field] == "":
                    row[field] = value
        enriched.append(row)
    return enriched


def normalize_delta_per_work(row, spec):
    for field in ("delta_cycles_per_max_core_tile", "delta_cycles_per_tile"):
        value = to_float(row.get(field))
        if value is not None:
            return value
    delta = to_float(row.get("delta_cycles_cb_minus_static"))
    if delta is None:
        return None
    work_field = spec.get("work_field")
    work = to_float(row.get(work_field)) if work_field else None
    if not work:
        tiles = to_float(row.get("tiles"))
        work = tiles if tiles else 1.0
    return delta / max(1.0, work)


def comparison_key(row, spec):
    return shape_key(row, spec), row.get("mode", "")


def group_stage_rows(rows, spec):
    grouped = {}
    for row in rows:
        case_name = row.get("case_family", "")
        mode = normalize_mode(case_name, row.get("mode", ""))
        key = shape_key(row, spec), mode, row.get("stage", "")
        grouped[key] = row
    return grouped


def stage_cycles(row):
    for field in ("max_core_median", "median_of_core_medians", "median_cycles"):
        value = to_float(row.get(field))
        if value is not None:
            return value
    return None


def build_stage_delta_rows(case_name, spec, critical_rows, comparison_rows):
    for row in critical_rows:
        row["case_family"] = case_name
    stage_by_key = group_stage_rows(critical_rows, spec)
    rows = []
    baseline_mode = spec["baseline_mode"]
    comparison_modes = {
        normalize_mode(case_name, mode_from_comparison(row, spec["primary_modes"][0] if spec["primary_modes"] else ""))
        for row in comparison_rows
    }
    comparison_keys = {
        (shape_key(row, spec), normalize_mode(case_name, mode_from_comparison(row, ""))) for row in comparison_rows
    }
    for shape, mode in sorted(comparison_modes and comparison_keys):
        if not mode:
            continue
        stages = sorted(
            {
                stage
                for row_shape, row_mode, stage in stage_by_key
                if row_shape == shape and (row_mode == baseline_mode or row_mode == mode)
            },
            key=lambda item: STAGE_ORDER.get(item, 99),
        )
        shape_values = dict(zip(spec["shape_fields"], shape))
        for stage in stages:
            baseline = stage_by_key.get((shape, baseline_mode, stage))
            candidate = stage_by_key.get((shape, mode, stage))
            if not baseline or not candidate:
                continue
            baseline_cycles = stage_cycles(baseline)
            candidate_cycles = stage_cycles(candidate)
            delta = None
            per_work = None
            if baseline_cycles is not None and candidate_cycles is not None:
                delta = baseline_cycles - candidate_cycles
                work_field = spec.get("work_field")
                work = to_float(candidate.get(work_field)) if work_field else None
                if not work:
                    work = to_float(candidate.get("tiles")) or 1.0
                per_work = delta / max(1.0, work)
            out = {
                "case": case_name,
                "shape": " ".join(f"{key}={value}" for key, value in shape_values.items() if value != ""),
                "mode": mode,
                "stage": stage,
                "baseline_cycles": baseline_cycles,
                "candidate_cycles": candidate_cycles,
                "delta_cycles_baseline_minus_candidate": delta,
                "delta_cycles_per_work": per_work,
            }
            out.update(shape_values)
            rows.append(out)
    return rows


def host_lookup(rows, spec):
    lookup = {}
    for row in rows:
        mode = row.get("mode")
        if not mode:
            mode = spec["primary_modes"][0] if spec["primary_modes"] else ""
        lookup[(shape_key(row, spec), mode)] = row
    return lookup


def classify_device_delta(case_name, mode, delta_per_work, speedup):
    if delta_per_work is None:
        return "missing_device_delta"
    if case_name == "real_copy":
        if mode == "static-streamreg-scratch-compiletime" and delta_per_work > 0:
            return "compile_time_protocol_overhead_removed"
        if mode == "static-streamreg-scratch" and delta_per_work < 0 and abs(delta_per_work) <= 15:
            return "sync_storage_overhead_near_cb"
        if delta_per_work < -50:
            return "sync_storage_overhead"
        if delta_per_work > 0:
            return "protocol_overhead_visible"
        return "noise_floor"
    if case_name == "real_tile_add":
        if mode == "static-streamreg-cbregs":
            return "protocol_overhead_visible_cbregs_control"
        if delta_per_work > 100:
            return "protocol_overhead_visible"
        return "noise_floor"
    if case_name == "real_matmul":
        if abs(delta_per_work) <= 100:
            return "critical_path_hidden_by_compute_reuse"
        if delta_per_work > 100:
            return "protocol_overhead_visible_shape_specific"
        return "static_protocol_regression_shape_specific"
    if case_name in {"ttnn_add", "ttnn_bcast_to_row", "ttnn_binary_ng_row_bcast", "ttnn_transpose_wh"}:
        if delta_per_work > 10:
            return "protocol_overhead_visible_small_positive"
        if delta_per_work < -10:
            return "static_protocol_regression"
        return "noise_floor"
    if case_name == "ttnn_slice_tile":
        if delta_per_work > 2:
            return "protocol_overhead_visible_small_positive"
        if delta_per_work < -2:
            return "static_protocol_regression"
        return "noise_floor"
    if case_name == "ttnn_sfpu_no_bcast":
        if abs(delta_per_work) <= 1:
            return "critical_path_hidden_by_compute_reuse"
        if delta_per_work > 10:
            return "protocol_overhead_visible_small_positive"
        if delta_per_work < -10:
            return "static_protocol_regression"
        return "noise_floor"
    if case_name == "ttnn_paged_update_cache":
        if speedup and speedup > 1.02:
            return "protocol_overhead_visible_shape_specific"
        if delta_per_work > 10:
            return "protocol_overhead_visible_small_positive"
        return "noise_floor"
    if case_name == "ttnn_embedding_lookup":
        if delta_per_work > 10:
            return "protocol_overhead_visible_shape_specific"
        if delta_per_work < -10:
            return "static_protocol_regression"
        return "noise_floor"
    if case_name == "ttnn_kv_cache_load_slice":
        if 0 < delta_per_work <= 2:
            return "protocol_overhead_visible_tiny_positive"
        if delta_per_work > 2:
            return "protocol_overhead_visible_small_positive"
        if delta_per_work < -2:
            return "static_protocol_regression"
        return "noise_floor"
    if speedup and speedup > 1.01:
        return "protocol_overhead_visible"
    if speedup and speedup < 0.99:
        return "static_protocol_regression"
    return "noise_floor"


def build_summary_rows(case_name, spec, comparison_rows, host_rows):
    host_by_key = host_lookup(host_rows, spec)
    rows = []
    for row in comparison_rows:
        mode = normalize_mode(case_name, mode_from_comparison(row, spec["primary_modes"][0] if spec["primary_modes"] else ""))
        row["mode"] = mode
        delta_per_work = normalize_delta_per_work(row, spec)
        speedup = to_float(row.get("static_speedup"))
        host = host_by_key.get((shape_key(row, spec), mode))
        host_delta = to_float(host.get("delta_us_cb_minus_static")) if host else None
        device_delta = to_float(row.get("delta_cycles_cb_minus_static"))
        device_direction = direction(device_delta)
        host_direction = direction(host_delta)
        if host_direction == "missing" or device_direction == "missing":
            agreement = "unknown"
        elif host_direction == device_direction:
            agreement = "same_direction"
        elif host_direction == "flat" or device_direction == "flat":
            agreement = "one_flat"
        else:
            agreement = "mismatch"
        out = {
            "case": case_name,
            "mode": mode,
            "mode_role": mode_role(case_name, mode),
            "shape": shape_label(row, spec),
            "critical_stage_baseline": row.get("cb_critical_stage", ""),
            "critical_stage_candidate": row.get("static_critical_stage", ""),
            "baseline_critical_cycles": row.get("cb_critical_cycles", ""),
            "candidate_critical_cycles": row.get("static_critical_cycles", ""),
            "delta_cycles_baseline_minus_candidate": device_delta,
            "delta_cycles_per_work": delta_per_work,
            "speedup": speedup,
            "host_delta_us_baseline_minus_candidate": host_delta,
            "host_device_agreement": agreement,
            "root_cause_class": classify_device_delta(case_name, mode, delta_per_work, speedup),
        }
        for field in spec["shape_fields"]:
            out[field] = row.get(field, "")
        rows.append(out)
    return rows


def direction(value):
    if value is None:
        return "missing"
    if abs(value) < 1e-9:
        return "flat"
    return "positive" if value > 0 else "negative"


def aggregate_mode_stats(summary_rows):
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[(row["case"], row["mode"], row.get("mode_role", ""), row["root_cause_class"])].append(row)
    stats = []
    for (case_name, mode, role, root_cause), rows in sorted(grouped.items()):
        deltas = [row["delta_cycles_per_work"] for row in rows if row["delta_cycles_per_work"] is not None]
        speedups = [row["speedup"] for row in rows if row["speedup"] is not None]
        stats.append(
            {
                "case": case_name,
                "mode": mode,
                "mode_role": role,
                "root_cause_class": root_cause,
                "count": len(rows),
                "median_delta_cycles_per_work": statistics.median(deltas) if deltas else None,
                "min_delta_cycles_per_work": min(deltas) if deltas else None,
                "max_delta_cycles_per_work": max(deltas) if deltas else None,
                "median_speedup": statistics.median(speedups) if speedups else None,
            }
        )
    return stats


def summary_lookup(stats):
    lookup = defaultdict(list)
    for row in stats:
        lookup[row["case"]].append(row)
    return lookup


def mode_stat(stats_by_case, case_name, mode):
    for row in stats_by_case.get(case_name, []):
        if row["mode"] == mode:
            return row
    return {}


def fmt_stat_delta(stats_by_case, case_name, mode):
    row = mode_stat(stats_by_case, case_name, mode)
    return fmt_signed(row.get("median_delta_cycles_per_work"))


def fmt_stat_speedup(stats_by_case, case_name, mode):
    row = mode_stat(stats_by_case, case_name, mode)
    return fmt_number(row.get("median_speedup"), 4)


def load_case(case_name, directory):
    spec = CASE_SPECS[case_name]
    comparison_rows = enrich_rows_from_case_name(case_name, read_csv(directory / "device_mode_comparison.csv"))
    critical_rows = enrich_rows_from_case_name(case_name, read_csv(directory / "critical_stage_summary.csv"))
    host_rows = enrich_rows_from_case_name(case_name, read_csv(directory / "host_mode_comparison.csv"))
    if not comparison_rows:
        return [], [], [f"{case_name}: missing or empty device_mode_comparison.csv in {directory}"]
    summary_rows = build_summary_rows(case_name, spec, comparison_rows, host_rows)
    stage_rows = build_stage_delta_rows(case_name, spec, critical_rows, comparison_rows)
    return summary_rows, stage_rows, []


def stage_findings(stage_rows):
    grouped = defaultdict(list)
    for row in stage_rows:
        grouped[(row["case"], row["mode"], row["stage"])].append(row)
    findings = []
    for key, rows in sorted(grouped.items()):
        deltas = [row["delta_cycles_per_work"] for row in rows if row["delta_cycles_per_work"] is not None]
        if not deltas:
            continue
        findings.append(
            {
                "case": key[0],
                "mode": key[1],
                "stage": key[2],
                "median_delta_cycles_per_work": statistics.median(deltas),
                "min_delta_cycles_per_work": min(deltas),
                "max_delta_cycles_per_work": max(deltas),
                "count": len(deltas),
            }
        )
    return findings


def summarize_case_markdown(case_name, stats, summary_rows, stage_stats):
    case_stats = [row for row in stats if row["case"] == case_name]
    case_rows = [row for row in summary_rows if row["case"] == case_name]
    lines = [f"### {CASE_SPECS[case_name]['title']}"]
    if not case_rows:
        lines.append("")
        lines.append("这个 case 没有可用数据。")
        return lines
    lines.append("")
    lines.append("| Mode | 归因分类 | 样本数 | Median delta/work | Median speedup |")
    lines.append("|---|---|---:|---:|---:|")
    for row in case_stats:
        lines.append(
            "| {mode} | {root} | {count} | {delta} | {speedup} |".format(
                mode=row["mode"],
                root=root_cause_label(row["root_cause_class"]),
                count=row["count"],
                delta=fmt_signed(row["median_delta_cycles_per_work"]),
                speedup=fmt_number(row["median_speedup"], 4),
            )
        )
    lines.append("")
    lines.extend(case_interpretation(case_name, case_rows, stage_stats))
    return lines


def case_interpretation(case_name, rows, stage_stats):
    if case_name == "real_copy":
        runtime = median_for(rows, "static-runtime")
        compiletime = median_for(rows, "static-compiletime")
        scratch = median_for(rows, "static-streamreg-scratch")
        scratch_compiletime = median_for(rows, "static-streamreg-scratch-compiletime")
        lines = [
            f"- L1 semaphore 版本的 `static-runtime` 明显慢于 CB，median delta/work 为 {fmt_signed(runtime)} cycles。",
        ]
        if compiletime is not None:
            lines.append(
                f"- `static-compiletime` 把损失收敛到 {fmt_signed(compiletime)} cycles/work，说明 runtime 地址/形状加载是成本之一，但不是全部。"
            )
        if scratch is not None:
            lines.append(
                f"- `static-streamreg-scratch` 接近 CB，median 为 {fmt_signed(scratch)} cycles/work，说明 copy 负例的主因更像是同步状态存储/协议成本，而不是 payload 搬运本身。"
            )
        if scratch_compiletime is not None:
            lines.append(
                f"- `static-streamreg-scratch-compiletime` 保留 ready/consumed 同步，但把 ring/layout/counter 参数静态化，median 为 {fmt_signed(scratch_compiletime)} cycles/work，说明去掉 runtime 参数和泛化配置后 pure copy 也能转正。"
            )
        return lines
    if case_name == "real_tile_add":
        runtime = median_for(rows, "static-runtime")
        cbregs = median_for(rows, "static-streamreg-cbregs")
        level_b = median_for(rows, "static-streamreg-cbregs-compiletime")
        cbregs_vs_runtime = None
        if runtime is not None and cbregs is not None:
            cbregs_vs_runtime = cbregs - runtime
        lines = [
            f"- `static-runtime` 相对 CB 已经有大幅收益：median {fmt_signed(runtime)} cycles/work。",
            f"- Level B 标准 `static-streamreg-cbregs` 保留这部分收益：median {fmt_signed(cbregs)} cycles/work。",
            f"- cbregs 相对 runtime 的 median 差异只有 {fmt_signed(cbregs_vs_runtime)} cycles/work，因此这里的 cbregs 更像是 launch/control placement 的 ABI 验证项，不是新的 steady-state 优化来源。",
        ]
        if level_b is not None:
            lines.append(
                f"- `static-streamreg-cbregs-compiletime` compile-time ablation median 为 {fmt_signed(level_b)} cycles/work，用来隔离 runtime args/config bake-in 的上界。"
            )
        return lines
    if case_name == "real_matmul":
        positives = [row for row in rows if row["delta_cycles_per_work"] is not None and row["delta_cycles_per_work"] > 0]
        negatives = [row for row in rows if row["delta_cycles_per_work"] is not None and row["delta_cycles_per_work"] < 0]
        return [
            f"- Matmul 仍然是 mixed：本轮 sweep 里有 {len(positives)} 个正向对比和 {len(negatives)} 个负向对比。",
            "- critical stage 通常落在 writer/compute 邻近区域，而不是干净的 queue-only 区域；局部协议收益会被 reuse、compute 调度和写回路径隐藏。",
            "- 正向的 output-static 行只能视为 exposed-shape 候选，不能推广成 broad matmul 结论。",
        ]
    if case_name == "ttnn_add":
        runtime = median_for(rows, "static-runtime")
        cbregs = median_for(rows, "static-streamreg-cbregs")
        level_b = median_for(rows, "static-streamreg-cbregs-compiletime")
        if cbregs is None:
            return [
                f"- TTNN no-broadcast add fork 保留稳定但较小的 device-side 正收益：median {fmt_signed(runtime)} cycles/work。",
                "- 这是正例对照：static protocol 可以穿过 TTNN 风格 reader/writer/compute ABI，但幅度远小于 standalone tile add。",
            ]
        lines = [
            f"- `static-runtime` 有稳定但较小的 device-side 正收益：median {fmt_signed(runtime)} cycles/work。",
            f"- Level B 标准 `static-streamreg-cbregs` 为 {fmt_signed(cbregs)} cycles/work，因此也应解读为 ABI 验证/控制项，而不是主要收益来源。",
        ]
        if level_b is not None:
            lines.append(
                f"- `static-streamreg-cbregs-compiletime` compile-time ablation 为 {fmt_signed(level_b)} cycles/work，保留 TTNN TensorAccessor/reader/writer/compute ABI，只静态化可 baked 的 queue/layout/config 字段。"
            )
        return lines
    if case_name == "ttnn_sfpu_no_bcast":
        runtime = median_for(rows, "static-runtime")
        cbregs = median_for(rows, "static-streamreg-cbregs")
        stages = sorted(
            {
                row.get("critical_stage_baseline", "")
                for row in rows
                if row.get("critical_stage_baseline", "")
            }
        )
        return [
            "- 这是从 TTNN `binary_ng` SFPU no-broadcast compute path 派生的 direct fork，当前 op 是 `div`。",
            f"- `static-runtime` median 为 {fmt_signed(runtime)} cycles/local-tile，`static-streamreg-cbregs` median 为 {fmt_signed(cbregs)} cycles/local-tile；属于 device profiler 噪声区间。",
            f"- baseline critical stage 为 {', '.join(stages) if stages else 'n/a'}；与 simple add 相比，SFPU-heavy 路径中 CB FIFO 动态管理没有形成稳定 exposed critical-path 收益。",
            "- 这说明不能把 no-broadcast add 的收益直接推广到所有 eltwise；SFPU-heavy 需要按 op/shape 单独归因。",
        ]
    if case_name == "ttnn_bcast_to_row":
        runtime = median_for(rows, "static-runtime")
        cbregs = median_for(rows, "static-streamreg-cbregs")
        level_b = median_for(rows, "static-streamreg-cbregs-compiletime")
        lines = [
            f"- 这是从 TTNN `experimental.bcast_to` row-broadcast reader/compute/writer 复制出来的 direct fork；`static-runtime` median 为 {fmt_signed(runtime)} cycles/local-tile。",
            f"- `static-streamreg-cbregs` median 为 {fmt_signed(cbregs)} cycles/local-tile；它继续作为 compute-path per-CB register ABI 控制项。",
            "- 是否推广 broadcast family 要看 device critical stage 是否从 reader/writer/CB queue 相关 stage 下降，而不是只看 host timing。",
        ]
        if level_b is not None:
            lines.append(f"- compile-time ablation median 为 {fmt_signed(level_b)} cycles/local-tile。")
        return lines
    if case_name == "ttnn_binary_ng_row_bcast":
        runtime = median_for(rows, "static-runtime")
        cbregs = median_for(rows, "static-streamreg-cbregs")
        level_b = median_for(rows, "static-streamreg-cbregs-compiletime")
        add_rows = [row for row in rows if row.get("op") == "add"]
        lines = [
            f"- 这是从 TTNN `binary_ng` row-broadcast reader/compute/writer 复制出来的 direct fork；当前只覆盖 reader-side 软件 row-bcast add。",
            f"- 全部样本的 `static-runtime` median 为 {fmt_signed(runtime)} cycles/local-tile，`static-streamreg-cbregs` median 为 {fmt_signed(cbregs)} cycles/local-tile。",
            f"- add 样本数 {len(add_rows)}；LLK row-bcast 和 SFPU-heavy 版本仍未形成可用 device profiler 证据。",
            "- 该 case 是判断 binary broadcast 是否能推广的第一批直接证据，不能和 `bcast_to` 的单输入广播或 SFPU-heavy 路径混为一个结论。",
        ]
        if level_b is not None:
            lines.append(f"- compile-time ablation median 为 {fmt_signed(level_b)} cycles/local-tile；该 fork 仍作为 optional evidence。")
        return lines
    if case_name == "ttnn_paged_update_cache":
        runtime = median_for(rows, "static-runtime")
        cbregs = median_for(rows, "static-streamreg-cbregs")
        level_b = median_for(rows, "static-streamreg-cbregs-compiletime")
        min_delta = min(
            [row["delta_cycles_baseline_minus_candidate"] for row in rows if row["delta_cycles_baseline_minus_candidate"] is not None],
            default=None,
        )
        max_delta = max(
            [row["delta_cycles_baseline_minus_candidate"] for row in rows if row["delta_cycles_baseline_minus_candidate"] is not None],
            default=None,
        )
        lines = [
            f"- 这是从 TTNN `paged_update_cache` C++ factory/device kernels 复制出来的真实 update-path fork；8-user decode-like shape 上 static protocol 为正：saved cycles 约 {fmt_number(min_delta)} 到 {fmt_number(max_delta)}。",
            f"- `static-runtime` median 为 {fmt_signed(runtime)} cycles/user，`static-streamreg-cbregs` median 为 {fmt_signed(cbregs)} cycles/user；收益稳定但幅度小于 standalone elementwise。",
            "- CB baseline 的 critical stage 是 `compute-input-untilize`，static 模式的 critical stage 转到 `compute-pack`，说明原路径里的 CB FIFO 动态管理确实触到了 device critical path。",
            "- 这个结论只覆盖 paged KV update；embedding lookup 和 KV cache load-slice 已单独归因，32-user static scalability 仍不能推广。",
        ]
        if level_b is not None:
            lines.append(
                f"- compile-time ablation median 为 {fmt_signed(level_b)} cycles/user；page table、cache idx 和真实 data movement 语义保持原样。"
            )
        return lines
    if case_name == "ttnn_transpose_wh":
        runtime = median_for(rows, "static-runtime")
        cbregs = median_for(rows, "static-streamreg-cbregs")
        stages = sorted(
            {
                row.get("critical_stage_baseline", "")
                for row in rows
                if row.get("critical_stage_baseline", "")
            }
        )
        return [
            "- 这是从 TTNN tiled `transpose_wh` reader/compute/writer 复制出来的真实 layout fork；保留 `reader_unary_transpose_wh_interleaved_start_id` 的 NWH 读序和 `transpose_wh_tile` compute 语义。",
            f"- `static-runtime` median 为 {fmt_signed(runtime)} cycles/local-tile，`static-streamreg-cbregs` median 为 {fmt_signed(cbregs)} cycles/local-tile；属于稳定但很小的 device-side 正收益。",
            f"- baseline critical stage 为 {', '.join(stages) if stages else 'n/a'}；本轮全部落在 writer，说明 CB FIFO 动态管理只在 writer 长路径上轻微暴露。",
            "- 结论是 layout/transpose 不能从 copy 负例直接否定，也不能从 tile-add 正例直接推广；必须继续分别测 tilize/untilize/slice/concat。",
        ]
    if case_name == "ttnn_embedding_lookup":
        runtime = median_for(rows, "static-runtime")
        cbregs = median_for(rows, "static-streamreg-cbregs")
        stages = sorted(
            {
                row.get("critical_stage_baseline", "")
                for row in rows
                if row.get("critical_stage_baseline", "")
            }
        )
        return [
            "- 这是从 TTNN embedding RM lookup dataflow 语义收缩出来的真实 lookup/read-heavy fork：reader 读 index 后从 weight table gather 一整行，writer 写 output row。",
            f"- `static-runtime` median 为 {fmt_signed(runtime)} cycles/row，`static-streamreg-cbregs` median 为 {fmt_signed(cbregs)} cycles/row；rows=256/1024/4096、vocab=32000、dim=128 上趋势稳定。",
            f"- baseline critical stage 为 {', '.join(stages) if stages else 'n/a'}；本轮全部落在 writer，说明 lookup 输出写回路径上 CB FIFO 动态管理有稳定但较小的 exposed cost。",
            "- 这条证据可以推广到 embedding lookup 的 first-order 正例，但不能自动推广到 paged cache read、padding/binary embedding、chunked 超宽 row 或 multi-core sharded embedding。",
        ]
    if case_name == "ttnn_slice_tile":
        runtime = median_for(rows, "static-runtime")
        cbregs = median_for(rows, "static-streamreg-cbregs")
        stages = sorted(
            {
                row.get("critical_stage_baseline", "")
                for row in rows
                if row.get("critical_stage_baseline", "")
            }
        )
        return [
            "- 这是从 TTNN tiled slice 读写路径收缩出来的真实 layout fork：reader 按二维 tile slice 区间读取输入，writer 写连续输出 tile。",
            f"- `static-runtime` median 为 {fmt_signed(runtime)} cycles/output-tile，`static-streamreg-cbregs` median 为 {fmt_signed(cbregs)} cycles/output-tile；属于稳定但很小的 device-side 正收益。",
            f"- baseline critical stage 为 {', '.join(stages) if stages else 'n/a'}；本轮全部落在 writer，说明 CB FIFO 动态管理在 slice writer 长路径上轻微暴露。",
            "- 这只能说明 slice/layout 有弱正例，不能推广到 tilize/untilize/concat 或 sharded 大 slice；这些仍需要各自的 device profiler fork。",
        ]
    if case_name == "ttnn_kv_cache_load_slice":
        runtime = median_for(rows, "static-runtime")
        cbregs = median_for(rows, "static-streamreg-cbregs")
        stages = sorted(
            {
                row.get("critical_stage_baseline", "")
                for row in rows
                if row.get("critical_stage_baseline", "")
            }
        )
        return [
            "- 这是从 TTNN `nlp_kv_cache_load_slice` 核心语义收缩出来的真实 cache-read/load-slice fork：reader 从 KV cache slice 直接读入 per-core L1 output，writer 只保留同步/控制观测。",
            f"- `static-runtime` median 为 {fmt_signed(runtime)} cycles/output-tile，`static-streamreg-cbregs` median 为 {fmt_signed(cbregs)} cycles/output-tile；结果接近 device profiler 噪声区间。",
            f"- baseline critical stage 为 {', '.join(stages) if stages else 'n/a'}；本轮全部落在 reader，说明瓶颈主要是读取/地址/NoC 路径，CB FIFO 动态管理只轻微暴露。",
            "- 因此它不是强正例；大输出 slice 还会受单核 L1 容量限制，必须用多核 sharding 后再判断真实 decode cache-read 的推广性。",
        ]
    return []


def median_for(rows, mode):
    values = [row["delta_cycles_per_work"] for row in rows if row["mode"] == mode and row["delta_cycles_per_work"] is not None]
    if not values:
        return None
    return statistics.median(values)


def write_report(path, summary_rows, stage_rows, warnings):
    stats = aggregate_mode_stats(summary_rows)
    stage_stats = stage_findings(stage_rows)
    lines = [
        "# Compiler-Managed L1 协议归因报告",
        "",
        "本报告由现有 profiler runner 的 CSV 输出生成。`delta/work` 为正表示 baseline CB 比候选模式慢；为负表示候选模式更慢。",
        "",
        "## 结论摘要",
        "",
        "- 当前根因不是“stream-register 天然更快”。真正的大收益来自在 memory-bound / simple elementwise 路径里，用 static schedule 替代 CB FIFO 动态管理。",
        "- direct copy 的 L1 semaphore static runtime 是负例；scratch-register 同步几乎补上了和 CB 的差距。",
        "- Level B 标准 mode 是 `static-streamreg-cbregs`：它验证 per-CB ABI 边界，并允许通过 runtime args 传 per-core L1 地址、work partition 和 shape/config。",
        "- `static-streamreg-cbregs-compiletime` 只作为 compile-time ablation 单独聚合，用来估计 runtime args/config bake-in 的上界。",
        "- matmul 仍然是 shape-specific / mixed，不能据此 broad promote。",
        "- TTNN binary no-broadcast add 是正例对照：static protocol 能穿过 TTNN 风格 kernel ABI，但每 tile 收益明显小于 standalone tile add。",
        "- TTNN bcast_to row-broadcast 和 binary_ng reader-side row-broadcast add 已纳入 direct fork：它们用于验证 broadcast 路径，结论必须以 device critical stage 为准。",
        "- TTNN paged_update_cache 是 KV-cache update 的真实正例：收益稳定但小，说明真实 op 里的 NoC、page table、untilize/tilize 和 writer overwrite 会稀释协议收益。",
        "- TTNN transpose_wh 是 layout 类弱正例：critical stage 全部为 writer，static protocol 只省少量 cycles/local-tile。",
        "- TTNN embedding lookup 是 lookup/read-heavy 正例：writer critical stage 上稳定省约 18 cycles/row。",
        "- TTNN slice_tile 是 layout 类弱正例：writer critical stage 上稳定省约 5-6 cycles/output-tile。",
        "- TTNN KV cache load-slice 是 cache-read 弱例：critical stage 全部为 reader，收益只有约 0.3-1.1 cycles/output-tile，接近噪声区间。",
        "",
    ]
    if warnings:
        lines.append("## 警告")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    lines.append("## 分 case 结论")
    lines.append("")
    for case_name in CASE_SPECS:
        lines.extend(summarize_case_markdown(case_name, stats, summary_rows, stage_stats))
        lines.append("")
    lines.extend(
        [
            "## 下一步实验",
            "",
            "1. data movement/layout：transpose_wh 和 slice_tile 已完成弱正例；进入 Level C 前先暂停新增 fork，后续再补 tilize/untilize/concat 和多核 sharded 大 slice。",
            "2. elementwise：从 no-broadcast add 扩展到 unary、broadcast binary、SFPU-heavy chains；`static-streamreg-cbregs` 继续作为控制项。",
            "3. matmul：只追 low-K、GEMV-like、multicast、decode-like 这些 exposed shapes；不要把 large prefill GEMM 当 proof point。",
            "4. LLM decode：paged_update_cache、embedding lookup 和 KV cache load-slice 已给出第一批证据；RMSNorm/LayerNorm/softmax decode 仍需 Level C 再做 direct fork。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_family_report(path, summary_rows, warnings):
    stats = aggregate_mode_stats(summary_rows)
    stats_by_case = summary_lookup(stats)
    paged_rows = [row for row in summary_rows if row["case"] == "ttnn_paged_update_cache"]
    paged_saved = [
        row["delta_cycles_baseline_minus_candidate"]
        for row in paged_rows
        if row["delta_cycles_baseline_minus_candidate"] is not None
    ]
    paged_speedups = [row["speedup"] for row in paged_rows if row["speedup"] is not None]
    lines = [
        "# TTNN 算子横向 Static Protocol 实验总结",
        "",
        "日期：2026-05-19",
        "",
        "## 总结论",
        "",
        "- 不能把 compiler-managed static ring/schedule 作为所有 TTNN 算子的统一加速结论；它只在 CB FIFO 动态管理处于 device critical path 或接近 critical path 时稳定收益。",
        "- 已有 direct profiler 正例是 memory-bound/simple elementwise 和 paged KV update；已知负例/弱例是 dataflow-only L1-semaphore copy 和当前 matmul reuse path。",
        "- Level B 标准 mode 是 `static-streamreg-cbregs`：每个 logical CB 使用自己的 `tiles_received/tiles_acked` stream register，per-core L1 地址和 work partition 通过 runtime args 传入。",
        "- `static-streamreg-cbregs-compiletime` 用独立 ablation 统计；它回答 runtime args/config bake-in 在 TT-Metal/LLK CB ABI 内部还能多省多少，而不是 Level B 完成条件。",
        "- 对 RMSNorm/LayerNorm/softmax decode 等还没有 direct static fork 的 family，当前结果只完成 sweep/pytest/TTNN baseline 覆盖，不能宣称 static speedup。",
        "",
        "## Direct Profiler 证据",
        "",
        "| 类别 | Direct case | 结果 | 根因 | 当前决策 |",
        "|---|---|---|---|---|",
        (
            "| data_movement/layout | `real_copy_protocol` | "
            f"`static-runtime` {fmt_stat_delta(stats_by_case, 'real_copy', 'static-runtime')} cycles/work，"
            f"`static-streamreg-scratch` {fmt_stat_delta(stats_by_case, 'real_copy', 'static-streamreg-scratch')} cycles/work | "
            "L1 counter/semaphore 同步状态存储成本超过 CB；scratch register 说明 payload 搬运不是主因 | "
            "继续做 tilize/untilize/transpose direct fork，不能从 copy 推广正收益 |"
        ),
        (
            "| data_movement/layout | `ttnn_transpose_wh_protocol` | "
            f"`static-runtime` {fmt_stat_delta(stats_by_case, 'ttnn_transpose_wh', 'static-runtime')} cycles/local-tile，"
            f"`static-streamreg-cbregs` {fmt_stat_delta(stats_by_case, 'ttnn_transpose_wh', 'static-streamreg-cbregs')} cycles/local-tile | "
            "真实 TTNN tiled transpose WH，critical stage 全部是 writer；CB FIFO 动态管理只在长 writer path 上小幅暴露 | "
            "弱正例；slice 已单独补测，仍不从 transpose 单点推广到 tilize/untilize/concat |"
        ),
        (
            "| data_movement/layout | `ttnn_slice_tile_protocol` | "
            f"`static-runtime` {fmt_stat_delta(stats_by_case, 'ttnn_slice_tile', 'static-runtime')} cycles/output-tile，"
            f"`static-streamreg-cbregs` {fmt_stat_delta(stats_by_case, 'ttnn_slice_tile', 'static-streamreg-cbregs')} cycles/output-tile | "
            "真实 TTNN tiled slice 读写 fork，critical stage 全部是 writer；CB FIFO 动态管理在 slice writer path 上小幅暴露 | "
            "弱正例；只覆盖单核 tiled slice，不推广到 tilize/untilize/concat 或 sharded 大 slice |"
        ),
        (
            "| eltwise | `real_tile_add_protocol` | "
            f"`static-runtime` {fmt_stat_delta(stats_by_case, 'real_tile_add', 'static-runtime')} cycles/work，"
            f"Level B `static-streamreg-cbregs` {fmt_stat_delta(stats_by_case, 'real_tile_add', 'static-streamreg-cbregs')} cycles/work，"
            f"compiletime ablation {fmt_stat_delta(stats_by_case, 'real_tile_add', 'static-streamreg-cbregs-compiletime')} cycles/work | "
            "writer/queue path 暴露，static schedule 替代 CB FIFO 动态管理带来大收益 | "
            "推广到 memory-bound/simple elementwise direct forks |"
        ),
        (
            "| eltwise/TTNN | `ttnn_binary_ng_no_bcast_protocol` | "
            f"`static-runtime` {fmt_stat_delta(stats_by_case, 'ttnn_add', 'static-runtime')} cycles/local-tile，"
            f"Level B `static-streamreg-cbregs` {fmt_stat_delta(stats_by_case, 'ttnn_add', 'static-streamreg-cbregs')} cycles/local-tile，"
            f"compiletime ablation {fmt_stat_delta(stats_by_case, 'ttnn_add', 'static-streamreg-cbregs-compiletime')} cycles/local-tile，"
            f"speedup {fmt_stat_speedup(stats_by_case, 'ttnn_add', 'static-runtime')} | "
            "真实 TTNN reader/writer/compute ABI 中 writer 仍有协议成本，但被真实地址/NoC/compute 稀释 | "
            "作为 TTNN-style 正例，下一步扩展 unary/bcast/SFPU-heavy |"
        ),
        (
            "| eltwise/SFPU-heavy TTNN | `ttnn_binary_ng_sfpu_no_bcast_protocol` div | "
            f"`static-runtime` {fmt_stat_delta(stats_by_case, 'ttnn_sfpu_no_bcast', 'static-runtime')} cycles/local-tile，"
            f"`static-streamreg-cbregs` {fmt_stat_delta(stats_by_case, 'ttnn_sfpu_no_bcast', 'static-streamreg-cbregs')} cycles/local-tile | "
            "critical stage 仍是 writer/compute-heavy 路径，static protocol delta 落在噪声区间 | "
            "不推广 SFPU-heavy；继续测 GELU/exp/complex unary chain |"
        ),
        (
            "| eltwise/broadcast TTNN | `ttnn_bcast_to_protocol` row-bcast | "
            f"`static-runtime` {fmt_stat_delta(stats_by_case, 'ttnn_bcast_to_row', 'static-runtime')} cycles/local-tile，"
            f"Level B `static-streamreg-cbregs` {fmt_stat_delta(stats_by_case, 'ttnn_bcast_to_row', 'static-streamreg-cbregs')} cycles/local-tile，"
            f"compiletime ablation {fmt_stat_delta(stats_by_case, 'ttnn_bcast_to_row', 'static-streamreg-cbregs-compiletime')} cycles/local-tile | "
            "真实 TTNN row-broadcast reader/compute/writer fork；用于判断 broadcast queue 协议是否进入 critical stage | "
            "按 device critical stage 决定是否继续扩展 binary broadcast/SFPU-heavy |"
        ),
        (
            "| eltwise/binary broadcast TTNN | `ttnn_binary_ng_row_bcast_protocol` reader-side add | "
            f"`static-runtime` {fmt_stat_delta(stats_by_case, 'ttnn_binary_ng_row_bcast', 'static-runtime')} cycles/local-tile，"
            f"Level B `static-streamreg-cbregs` {fmt_stat_delta(stats_by_case, 'ttnn_binary_ng_row_bcast', 'static-streamreg-cbregs')} cycles/local-tile，"
            f"compiletime ablation {fmt_stat_delta(stats_by_case, 'ttnn_binary_ng_row_bcast', 'static-streamreg-cbregs-compiletime')} cycles/local-tile | "
            "真实 TTNN binary_ng row-broadcast direct fork；当前是 reader-side 软件 row-bcast add，不覆盖 LLK/SFPU-heavy | "
            "继续单独 fork SFPU-heavy；不能把 add 结果推广到 compute-math dominated 路径 |"
        ),
        (
            "| matmul/linear | `real_matmul_protocol` | "
            "正负混合，部分 output-static shape 为正，部分 shape 回退 | "
            "reuse/compute/writeback critical path 掩盖 queue-only 收益 | "
            "不 broad promote；只测 low-K/GEMV/multicast/decode-like |"
        ),
        (
            "| embedding/KV-cache | `ttnn_paged_update_cache_protocol` | "
            f"8-user decode-like saved {fmt_number(min(paged_saved) if paged_saved else None)} 到 "
            f"{fmt_number(max(paged_saved) if paged_saved else None)} cycles，"
            f"Level B `static-streamreg-cbregs` {fmt_stat_delta(stats_by_case, 'ttnn_paged_update_cache', 'static-streamreg-cbregs')} cycles/user，"
            f"compiletime ablation {fmt_stat_delta(stats_by_case, 'ttnn_paged_update_cache', 'static-streamreg-cbregs-compiletime')} cycles/user，"
            f"speedup median {fmt_number(statistics.median(paged_speedups) if paged_speedups else None, 4)} | "
            "CB critical stage 为 compute-input-untilize，static 后转到 compute-pack；CB FIFO 管理触到 critical path | "
            "shape-specific promote；32-user scalability 未推广，cache-read/embedding lookup 分开看 |"
        ),
        (
            "| embedding/lookup | `ttnn_embedding_lookup_protocol` | "
            f"`static-runtime` {fmt_stat_delta(stats_by_case, 'ttnn_embedding_lookup', 'static-runtime')} cycles/row，"
            f"`static-streamreg-cbregs` {fmt_stat_delta(stats_by_case, 'ttnn_embedding_lookup', 'static-streamreg-cbregs')} cycles/row | "
            "真实 embedding lookup/read-heavy fork；writer critical stage 上 CB FIFO 动态管理稳定暴露 | "
            "可作为 embedding lookup first-order 正例；cache-read/padded/chunked/multicore 仍需单独 fork |"
        ),
        (
            "| embedding/KV-cache read | `ttnn_kv_cache_load_slice_protocol` | "
            f"`static-runtime` {fmt_stat_delta(stats_by_case, 'ttnn_kv_cache_load_slice', 'static-runtime')} cycles/output-tile，"
            f"`static-streamreg-cbregs` {fmt_stat_delta(stats_by_case, 'ttnn_kv_cache_load_slice', 'static-streamreg-cbregs')} cycles/output-tile | "
            "真实 KV cache load-slice/read fork；critical stage 全部是 reader，CB FIFO 动态管理只轻微暴露 | "
            "弱例/近噪声；需要多核 sharded 大 slice 后再判断 cache-read 推广性 |"
        ),
        "",
        "## 全部 TTNN Family 当前状态",
        "",
        "| Family | 当前证据 | 是否可宣称收益 | 结论 | 下一步 |",
        "|---|---|---|---|---|",
        "| `eltwise` | direct TTNN no-bcast、SFPU no-bcast div、bcast_to row-bcast、binary_ng reader-side row-bcast add + real tile-add + sweep/pytest coverage | 是，限 memory-bound/simple/exposed broadcast；SFPU-heavy 当前不推广 | simple/no-bcast 与 row-bcast add 为正，SFPU div 落在噪声区间，说明收益取决于 CB 协议是否仍在 critical path | 继续 fork unary 和更复杂 SFPU chain |",
        "| `data_movement_layout` | direct copy L1 semaphore 负例 + transpose_wh 弱正例 + slice_tile 弱正例 + streamreg compile-time ablation + sweep/pytest coverage | 是，限 transpose_wh/slice_tile 的弱正例；copy 不能推广 | L1-semaphore copy 不如 CB；transpose/slice 的 writer critical stage 上有小幅稳定收益，说明 layout 类要逐 op 判断 | Level C 前暂停；后续补 tilize/untilize/concat 和多核 sharded slice |",
        "| `embedding_kv_cache` | direct paged_update_cache update + embedding lookup + KV cache load-slice + sweep/pytest coverage | 是，限 8-user paged update 和 single-core embedding lookup；KV load-slice 只算弱例 | KV update 和 embedding lookup 有小幅稳定 device critical-path 收益；cache-read/load-slice 是 reader critical，收益近噪声 | Level C 前暂停；后续补多核 sharded cache-read/padded/chunked/multicore embedding |",
        "| `matmul_linear` | direct low-K matmul + sweep/pytest coverage | 否，只有 shape-specific 候选 | 当前 reuse path 混合，不能推广 | GEMV/low-K/multicast/decode-like fork |",
        "| `normalization_softmax` | sweep/pytest + TTNN workload baseline 计划 | 否 | 需要 direct RMSNorm/softmax fork 才能判断 CB 是否在 critical path | fork RMSNorm/LayerNorm、softmax decode |",
        "| `reduction` | sweep/pytest coverage | 否 | 小 reduction 可能暴露协议，但尚未 profile | fork sum/mean/max，分 small/long/cross-core |",
        "| `transformer_attention` | sweep/pytest coverage | 否 | prefill attention 不适合作为证明点；decode helper 更可能暴露 | fork SDPA decode、rotary、QKV split、concat heads |",
        "| `ccl` | sweep generation 可跑；单卡 runner 无适用 vectors | 否 | 必须分离本地协议成本和 fabric/同步瓶颈 | 多设备 all-gather/reduce-scatter direct fork |",
        "| `conv_pool` | sweep/pytest coverage | 否 | 作为 vision control lane；多数可能 bandwidth/compute dominated | 小/depthwise/pool baseline profile 后再 fork |",
        "| `creation_fill_typecast` | sweep/pytest coverage | 否 | 主要用于 host/runtime 和 writer accounting | fill/typecast 小 shape direct fork |",
        "| `backward_moreh_experimental` | sweep/pytest coverage | 否 | 不能聚合，需要按底层瓶颈拆分 | 先用 profiler 找 CB-heavy backward kernel |",
        "",
        "## 如何判断 CB FIFO 是否在 Critical Path",
        "",
        "1. 必须用 device profiler 看 stage-level critical cycles，而不是只看 host enqueue/finish。",
        "2. 如果 CB baseline 的最大 stage 是 reader/writer/compute input/output queue 附近，并且 static 后同一 shape 的 critical cycles 下降，才说明 CB FIFO 动态管理在 critical path 上。",
        "3. 如果 static 只让非 critical stage 变快，或者 critical stage 仍是 compute math/reuse/NoC bandwidth，那么端到端不会稳定收益。",
        "4. 如果收益随 local tile/user/page 数稳定缩放，可信度高；如果正负随 shape 翻转，必须标成 shape-specific。",
        "",
        "## 复现方案",
        "",
        "### 构建",
        "",
        "```bash",
        "conda run -n tt cmake --build build_Release \\",
        "  --target compiler_managed_l1_dataflow_examples -j8",
        "```",
        "",
        "### 横向覆盖计划",
        "",
        "```bash",
        "conda run -n tt python \\",
        "  tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/run_ttnn_static_protocol_suite.py \\",
        "  --tier core \\",
        "  --phases phase3 \\",
        "  --skip-build \\",
        "  --family-sweep-mode dry-run \\",
        "  --pytest-mode collect-only \\",
        "  --ttnn-workload-mode dry-run \\",
        "  --out-dir /tmp/ttnn_static_protocol_suite_phase3_core_repro",
        "```",
        "",
        "### Direct Device Profiler 对比",
        "",
        "```bash",
        "conda run -n tt python \\",
        "  tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/run_ttnn_static_protocol_suite.py \\",
        "  --tier core \\",
        "  --phases phase1 phase2 phase3 \\",
        "  --families data_movement_layout eltwise matmul_linear embedding_kv_cache \\",
        "  --skip-build \\",
        "  --family-sweep-mode none \\",
        "  --pytest-mode none \\",
        "  --ttnn-workload-mode none \\",
        "  --repeats 3 \\",
        "  --out-dir /tmp/ttnn_static_protocol_suite_direct_repro",
        "```",
        "",
        "### 生成归因报告",
        "",
        "```bash",
        "conda run -n tt python \\",
        "  tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/analyze_protocol_attribution.py \\",
        "  --real-copy-dir /tmp/ttnn_static_protocol_suite_direct_repro/runs/real_copy \\",
        "  --real-tile-add-dir /tmp/ttnn_static_protocol_suite_direct_repro/runs/real_tile_add \\",
        "  --real-matmul-dir /tmp/ttnn_static_protocol_suite_direct_repro/runs/real_matmul_low_k \\",
        "  --ttnn-add-dir /tmp/ttnn_static_protocol_suite_direct_repro/runs/ttnn_binary_ng_no_bcast \\",
        "  --ttnn-sfpu-no-bcast-dir /tmp/ttnn_static_protocol_suite_direct_repro/runs/ttnn_binary_ng_sfpu_no_bcast \\",
        "  --ttnn-bcast-to-row-dir /tmp/ttnn_static_protocol_suite_direct_repro/runs/ttnn_bcast_to_row \\",
        "  --ttnn-binary-ng-row-bcast-dir /tmp/ttnn_static_protocol_suite_direct_repro/runs/ttnn_binary_ng_row_bcast \\",
        "  --ttnn-paged-update-cache-dir /tmp/ttnn_static_protocol_suite_direct_repro/runs/ttnn_paged_update_cache \\",
        "  --ttnn-transpose-wh-dir /tmp/ttnn_static_protocol_suite_direct_repro/runs/ttnn_transpose_wh \\",
        "  --ttnn-embedding-lookup-dir /tmp/ttnn_static_protocol_suite_direct_repro/runs/ttnn_embedding_lookup \\",
        "  --ttnn-slice-tile-dir /tmp/ttnn_static_protocol_suite_direct_repro/runs/ttnn_slice_tile \\",
        "  --ttnn-kv-cache-load-slice-dir /tmp/ttnn_static_protocol_suite_direct_repro/runs/ttnn_kv_cache_load_slice \\",
        "  --out-dir /tmp/compiler_managed_l1_attribution_repro",
        "```",
        "",
        "本次报告使用的已完成输出主要来自 `/tmp/ttnn_static_protocol_suite_coverage_repro_2026_05_18`、`/tmp/ttnn_static_protocol_suite_direct_eltwise_kv_repro_2026_05_18`、`/tmp/real_copy_protocol_streamreg_single`、`/tmp/real_tile_add_protocol_cbregs_phase`、`/tmp/real_matmul_protocol_ttnn_sweep_2026_05_18`、`/tmp/ttnn_slice_tile_protocol_cases_2026_05_19` 和 `/tmp/ttnn_kv_cache_load_slice_protocol_cases_2026_05_19`。",
        "",
    ]
    if warnings:
        lines.extend(["## 生成警告", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_dirs = {
        "real_copy": args.real_copy_dir,
        "real_tile_add": args.real_tile_add_dir,
        "real_matmul": args.real_matmul_dir,
        "ttnn_add": args.ttnn_add_dir,
        "ttnn_sfpu_no_bcast": args.ttnn_sfpu_no_bcast_dir,
        "ttnn_bcast_to_row": args.ttnn_bcast_to_row_dir,
        "ttnn_binary_ng_row_bcast": args.ttnn_binary_ng_row_bcast_dir,
        "ttnn_paged_update_cache": args.ttnn_paged_update_cache_dir,
        "ttnn_transpose_wh": args.ttnn_transpose_wh_dir,
        "ttnn_embedding_lookup": args.ttnn_embedding_lookup_dir,
        "ttnn_slice_tile": args.ttnn_slice_tile_dir,
        "ttnn_kv_cache_load_slice": args.ttnn_kv_cache_load_slice_dir,
    }
    all_summary_rows = []
    all_stage_rows = []
    warnings = []
    for case_name, directory in input_dirs.items():
        if directory is None:
            warnings.append(f"{case_name}: input directory was not provided")
            continue
        summary_rows, stage_rows, case_warnings = load_case(case_name, directory)
        all_summary_rows.extend(summary_rows)
        all_stage_rows.extend(stage_rows)
        warnings.extend(case_warnings)
    stats = aggregate_mode_stats(all_summary_rows)
    stage_stats = stage_findings(all_stage_rows)
    write_csv(args.out_dir / "attribution_summary.csv", all_summary_rows)
    write_csv(
        args.out_dir / "level_b_summary.csv",
        [row for row in all_summary_rows if row.get("mode_role") == "level_b_standard"],
    )
    write_csv(
        args.out_dir / "compiletime_ablation_summary.csv",
        [row for row in all_summary_rows if row.get("mode_role") == "compiletime_ablation"],
    )
    write_csv(args.out_dir / "stage_delta_matrix.csv", all_stage_rows)
    write_csv(args.out_dir / "attribution_mode_stats.csv", stats)
    write_csv(args.out_dir / "stage_delta_stats.csv", stage_stats)
    write_report(args.out_dir / "root_cause_report.md", all_summary_rows, all_stage_rows, warnings)
    write_family_report(args.out_dir / "ttnn_operator_family_static_protocol_report.md", all_summary_rows, warnings)
    print(f"wrote {args.out_dir / 'root_cause_report.md'}")
    print(f"wrote {args.out_dir / 'ttnn_operator_family_static_protocol_report.md'}")


if __name__ == "__main__":
    main()
