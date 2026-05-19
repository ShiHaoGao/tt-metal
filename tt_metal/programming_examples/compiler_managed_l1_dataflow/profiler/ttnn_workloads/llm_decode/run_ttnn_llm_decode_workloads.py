#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import datetime as dt
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[6]


@dataclass(frozen=True)
class WorkloadSpec:
    case_id: str
    family: str
    workload: str
    mode: str
    shape_label: str
    fields: dict[str, str]
    logical_work_items: int
    notes: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run real TTNN LLM decode-like workloads as a baseline selector for compiler-managed L1/static-protocol forks."
        )
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Only write the workload matrix; do not open a device.")
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["rmsnorm", "softmax-decode", "paged-update-cache"],
        default=["rmsnorm", "softmax-decode", "paged-update-cache"],
    )
    parser.add_argument("--rmsnorm-seq-lens", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--rmsnorm-hidden-dims", nargs="+", type=int, default=[8192])
    parser.add_argument("--rmsnorm-epsilon", type=float, default=1.0e-6)
    parser.add_argument(
        "--rmsnorm-layout",
        choices=["width-sharded", "interleaved"],
        default="width-sharded",
        help="Default to width-sharded because large decode hidden dims can exceed single-core L1 in the default path.",
    )
    parser.add_argument("--rmsnorm-grid-x", type=int, default=8)
    parser.add_argument("--rmsnorm-grid-y", type=int, default=8)
    parser.add_argument("--softmax-heads", nargs="+", type=int, default=[64])
    parser.add_argument("--softmax-kv-tokens", nargs="+", type=int, default=[128, 1024, 4096, 8192])
    parser.add_argument("--paged-users", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--paged-kv-heads", nargs="+", type=int, default=[8])
    parser.add_argument("--paged-head-dims", nargs="+", type=int, default=[128])
    parser.add_argument("--paged-block-sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--paged-max-seq-lens", nargs="+", type=int, default=[2048])
    parser.add_argument("--paged-cache-idx", type=int, default=127)
    parser.add_argument("--paged-input-dtype", choices=["bfloat16", "bfloat8_b"], default="bfloat16")
    parser.add_argument("--paged-cache-dtype", choices=["bfloat16", "bfloat8_b"], default="bfloat16")
    parser.add_argument(
        "--shuffle-page-table",
        action="store_true",
        help="Shuffle physical pages during setup so the page table is non-identity.",
    )
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first workload failure.")
    return parser.parse_args()


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def spec_rows(specs):
    rows = []
    for spec in specs:
        row = {
            "case_id": spec.case_id,
            "family": spec.family,
            "workload": spec.workload,
            "mode": spec.mode,
            "shape_label": spec.shape_label,
            "logical_work_items": spec.logical_work_items,
            "notes": spec.notes,
        }
        row.update(spec.fields)
        rows.append(row)
    return rows


def build_workload_specs(args):
    specs = []
    if "rmsnorm" in args.workloads:
        for seq_len in args.rmsnorm_seq_lens:
            for hidden_dim in args.rmsnorm_hidden_dims:
                if hidden_dim % 32 != 0:
                    raise RuntimeError(f"RMSNorm hidden dim must be divisible by 32: {hidden_dim}")
                specs.append(
                    WorkloadSpec(
                        case_id=f"rmsnorm_s{seq_len}_h{hidden_dim}",
                        family="normalization_softmax",
                        workload="rmsnorm",
                        mode="ttnn-baseline",
                        shape_label=f"B=1 S={seq_len} hidden={hidden_dim}",
                        fields={
                            "batch": "1",
                            "seq_len": str(seq_len),
                            "hidden_dim": str(hidden_dim),
                            "heads": "",
                            "kv_tokens": "",
                            "paged_users": "",
                            "kv_heads": "",
                            "head_dim": "",
                            "block_size": "",
                            "max_seq_len": "",
                            "tokens_per_user": "",
                        },
                        logical_work_items=seq_len,
                        notes="Llama-style decode/prefill RMSNorm shape [1,1,S,hidden].",
                    )
                )
    if "softmax-decode" in args.workloads:
        for heads in args.softmax_heads:
            for kv_tokens in args.softmax_kv_tokens:
                if kv_tokens % 32 != 0:
                    raise RuntimeError(f"Softmax kv_tokens must be divisible by 32: {kv_tokens}")
                specs.append(
                    WorkloadSpec(
                        case_id=f"softmax_decode_h{heads}_kv{kv_tokens}",
                        family="normalization_softmax",
                        workload="softmax-decode",
                        mode="ttnn-baseline",
                        shape_label=f"B=1 heads={heads} q=1 kv={kv_tokens}",
                        fields={
                            "batch": "1",
                            "seq_len": "",
                            "hidden_dim": "",
                            "heads": str(heads),
                            "kv_tokens": str(kv_tokens),
                            "paged_users": "",
                            "kv_heads": "",
                            "head_dim": "",
                            "block_size": "",
                            "max_seq_len": "",
                            "tokens_per_user": "",
                        },
                        logical_work_items=heads,
                        notes="Decode attention softmax shape [1,heads,1,kv_tokens].",
                    )
                )
    if "paged-update-cache" in args.workloads:
        for users in args.paged_users:
            for kv_heads in args.paged_kv_heads:
                for head_dim in args.paged_head_dims:
                    for block_size in args.paged_block_sizes:
                        for max_seq_len in args.paged_max_seq_lens:
                            if kv_heads > 32:
                                raise RuntimeError(f"Paged update cache kv_heads must be <= 32: {kv_heads}")
                            if head_dim % 32 != 0:
                                raise RuntimeError(f"Paged update cache head_dim must be divisible by 32: {head_dim}")
                            if max_seq_len % block_size != 0:
                                raise RuntimeError(
                                    f"Paged update cache max_seq_len must be divisible by block_size: {max_seq_len}, {block_size}"
                                )
                            specs.append(
                                WorkloadSpec(
                                    case_id=(
                                        f"paged_update_u{users}_kvh{kv_heads}_d{head_dim}"
                                        f"_b{block_size}_s{max_seq_len}"
                                    ),
                                    family="embedding_kv_cache",
                                    workload="paged-update-cache",
                                    mode="ttnn-baseline",
                                    shape_label=(
                                        f"users={users} kv_heads={kv_heads} head_dim={head_dim} "
                                        f"block={block_size} max_seq={max_seq_len}"
                                    ),
                                    fields={
                                        "batch": "1",
                                        "seq_len": "",
                                        "hidden_dim": "",
                                        "heads": "",
                                        "kv_tokens": "",
                                        "paged_users": str(users),
                                        "kv_heads": str(kv_heads),
                                        "head_dim": str(head_dim),
                                        "block_size": str(block_size),
                                        "max_seq_len": str(max_seq_len),
                                        "tokens_per_user": "1",
                                    },
                                    logical_work_items=users,
                                    notes="Paged KV decode update, one token per active user per call.",
                                )
                            )
    return specs


def ttnn_dtype(ttnn, name):
    mapping = {
        "bfloat16": ttnn.bfloat16,
        "bfloat8_b": ttnn.bfloat8_b,
    }
    return mapping[name]


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def choose_grid(device, requested_x, requested_y, max_shards):
    compute_grid = device.compute_with_storage_grid_size()
    grid_x = max(1, min(int(compute_grid.x), requested_x, max_shards))
    grid_y = max(1, min(int(compute_grid.y), requested_y, max(1, max_shards // grid_x)))
    while grid_x * grid_y > max_shards and grid_y > 1:
        grid_y -= 1
    while grid_x * grid_y > max_shards and grid_x > 1:
        grid_x -= 1
    return grid_x, grid_y


def setup_rmsnorm_case(spec, device, torch, ttnn, args):
    seq_len = int(spec.fields["seq_len"])
    hidden_dim = int(spec.fields["hidden_dim"])
    torch_input = torch.rand((1, 1, seq_len, hidden_dim), dtype=torch.bfloat16)
    torch_weight = torch.rand((1, 1, 1, hidden_dim), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if hasattr(ttnn, "fill_implicit_tile_padding"):
        input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, 0.0)
    weight = ttnn.from_torch(
        torch_weight.reshape(1, 1, hidden_dim // 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if args.rmsnorm_layout == "width-sharded":
        max_shards = max(1, hidden_dim // 32)
        grid_x, grid_y = choose_grid(device, args.rmsnorm_grid_x, args.rmsnorm_grid_y, max_shards)
        num_shards = grid_x * grid_y
        padded_rows = ceil_div(seq_len, 32) * 32
        shard_width = ceil_div(ceil_div(hidden_dim, num_shards), 32) * 32
        shard_shape = [padded_rows, shard_width]
        input_tensor = ttnn.interleaved_to_sharded(
            input_tensor,
            (grid_x, grid_y),
            shard_shape,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        memory_config = ttnn.get_memory_config(input_tensor)
        block_w = max(1, shard_width // 32)
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            subblock_w=min(4, block_w),
            block_h=max(1, padded_rows // 32),
            block_w=block_w,
            inplace=False,
        )
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
        )

        def run_once():
            return ttnn.rms_norm(
                input_tensor,
                weight=weight,
                epsilon=args.rmsnorm_epsilon,
                memory_config=memory_config,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )

        return run_once

    def run_once():
        return ttnn.rms_norm(input_tensor, weight=weight, epsilon=args.rmsnorm_epsilon)

    return run_once


def setup_softmax_decode_case(spec, device, torch, ttnn, _args):
    heads = int(spec.fields["heads"])
    kv_tokens = int(spec.fields["kv_tokens"])
    torch_input = torch.randn((1, heads, 1, kv_tokens), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def run_once():
        return ttnn.softmax(input_tensor, dim=-1, numeric_stable=True)

    return run_once


def setup_paged_update_cache_case(spec, device, torch, ttnn, args):
    users = int(spec.fields["paged_users"])
    kv_heads = int(spec.fields["kv_heads"])
    head_dim = int(spec.fields["head_dim"])
    block_size = int(spec.fields["block_size"])
    max_seq_len = int(spec.fields["max_seq_len"])
    max_num_blocks_per_seq = max_seq_len // block_size
    max_num_blocks = users * max_num_blocks_per_seq

    cache_dtype = ttnn_dtype(ttnn, args.paged_cache_dtype)
    input_dtype = ttnn_dtype(ttnn, args.paged_input_dtype)

    paged_cache = torch.randn((max_num_blocks, kv_heads, block_size, head_dim), dtype=torch.bfloat16)
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(users, max_num_blocks_per_seq)
    if args.shuffle_page_table:
        permutation = torch.randperm(max_num_blocks)
        reverse_permutation = torch.argsort(permutation).to(torch.int32)
        paged_cache = paged_cache[permutation].contiguous()
        page_table = reverse_permutation.reshape(users, max_num_blocks_per_seq)

    cache_source = paged_cache.to(torch.float32) if args.paged_cache_dtype != "bfloat16" else paged_cache
    cache_tensor = ttnn.Tensor(cache_source, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)

    x = torch.randn((1, users, kv_heads, head_dim), dtype=torch.bfloat16)
    x_pad = torch.nn.functional.pad(x, (0, 0, 0, 32 - kv_heads), "constant", 0)
    input_tensor = ttnn.Tensor(x_pad, input_dtype).to(ttnn.TILE_LAYOUT)

    compute_grid_size = device.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(users, compute_grid_size, True)
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        [
            input_tensor.volume() // input_tensor.padded_shape[-1] // users,
            input_tensor.padded_shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    input_tensor = input_tensor.to(device, input_mem_config)

    update_idxs = [(args.paged_cache_idx + user * 17) % max_seq_len for user in range(users)]
    update_idxs_tensor = ttnn.Tensor(torch.tensor(update_idxs, dtype=torch.int32), ttnn.int32).to(device)
    page_table_tensor = ttnn.Tensor(page_table, ttnn.int32).to(device)

    def run_once():
        nonlocal cache_tensor
        cache_tensor = ttnn.experimental.paged_update_cache(
            cache_tensor,
            input_tensor,
            update_idxs_tensor=update_idxs_tensor,
            page_table=page_table_tensor,
            block_size=block_size,
        )
        return cache_tensor

    return run_once


def setup_case(spec, device, torch, ttnn, args):
    if spec.workload == "rmsnorm":
        return setup_rmsnorm_case(spec, device, torch, ttnn, args)
    if spec.workload == "softmax-decode":
        return setup_softmax_decode_case(spec, device, torch, ttnn, args)
    if spec.workload == "paged-update-cache":
        return setup_paged_update_cache_case(spec, device, torch, ttnn, args)
    raise RuntimeError(f"Unsupported workload: {spec.workload}")


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def fmt_float(value, digits=3):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def measure_case(spec, device, torch, ttnn, args):
    run_once = setup_case(spec, device, torch, ttnn, args)
    for _ in range(args.warmup):
        run_once()
        ttnn.synchronize_device(device)

    raw_rows = []
    times_us = []
    for repeat in range(args.repeats):
        start_ns = time.perf_counter_ns()
        run_once()
        ttnn.synchronize_device(device)
        elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
        times_us.append(elapsed_us)
        raw_row = {
            "case_id": spec.case_id,
            "family": spec.family,
            "workload": spec.workload,
            "mode": spec.mode,
            "shape_label": spec.shape_label,
            "repeat": repeat,
            "elapsed_us": f"{elapsed_us:.3f}",
            "logical_work_items": spec.logical_work_items,
            "us_per_work_item": f"{elapsed_us / max(1, spec.logical_work_items):.6f}",
        }
        raw_row.update(spec.fields)
        raw_rows.append(raw_row)

    median_us = statistics.median(times_us)
    summary = {
        "status": "pass",
        "case_id": spec.case_id,
        "family": spec.family,
        "workload": spec.workload,
        "mode": spec.mode,
        "shape_label": spec.shape_label,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "logical_work_items": spec.logical_work_items,
        "median_us": f"{median_us:.3f}",
        "mean_us": f"{statistics.mean(times_us):.3f}",
        "min_us": f"{min(times_us):.3f}",
        "max_us": f"{max(times_us):.3f}",
        "p10_us": fmt_float(percentile(times_us, 0.10)),
        "p90_us": fmt_float(percentile(times_us, 0.90)),
        "stdev_us": fmt_float(statistics.stdev(times_us) if len(times_us) > 1 else 0.0),
        "median_us_per_work_item": f"{median_us / max(1, spec.logical_work_items):.6f}",
        "notes": spec.notes,
        "error": "",
    }
    summary.update(spec.fields)
    return raw_rows, summary


def failed_case_summary(spec, args, error):
    row = {
        "status": "fail",
        "case_id": spec.case_id,
        "family": spec.family,
        "workload": spec.workload,
        "mode": spec.mode,
        "shape_label": spec.shape_label,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "logical_work_items": spec.logical_work_items,
        "median_us": "",
        "mean_us": "",
        "min_us": "",
        "max_us": "",
        "p10_us": "",
        "p90_us": "",
        "stdev_us": "",
        "median_us_per_work_item": "",
        "notes": spec.notes,
        "error": str(error).splitlines()[0][:500],
    }
    row.update(spec.fields)
    return row


def priority_bucket(row):
    if row.get("status") == "fail":
        return "failed"
    workload = row["workload"]
    median_us = float(row["median_us"])
    per_item = float(row["median_us_per_work_item"])
    if workload == "paged-update-cache":
        if per_item >= 15.0:
            return "high"
        return "medium"
    if workload == "rmsnorm":
        if row.get("seq_len") == "1" and median_us >= 40.0:
            return "high"
        return "medium"
    if workload == "softmax-decode":
        if int(row.get("kv_tokens") or 0) <= 1024 and median_us >= 40.0:
            return "high"
        return "medium"
    return "medium"


def render_report(args, specs, summaries, path):
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    lines = [
        "# TTNN LLM Decode 真实 Workload Baseline",
        "",
        f"Generated: `{generated}`",
        f"Out dir: `{args.out_dir}`",
        f"Mode: `{'dry-run' if args.dry_run else 'ttnn-baseline'}`",
        "",
        "## 目的",
        "",
        "这个 runner 的目的不是证明 static protocol 已经有收益，而是先用真实 TTNN op 和真实 decode-like shape 找下一批直接 C++ static fork 的优先级。",
        "因此这里的 timing 是 `op enqueue + ttnn.synchronize_device(device)` 的 host end-to-end latency，不等价于 device critical path，也不能直接归因到 CB FIFO。",
        "",
        "## Workload Matrix",
        "",
        "| Case | Workload | Shape | Logical work | Notes |",
        "|---|---|---|---:|---|",
    ]
    for spec in specs:
        lines.append(
            f"| `{spec.case_id}` | `{spec.workload}` | {spec.shape_label} | {spec.logical_work_items} | {spec.notes} |"
        )

    if summaries:
        lines.extend(
            [
                "",
                "## Host Timing Summary",
                "",
                "| Case | Status | Workload | Shape | Median us | us/work | Priority |",
                "|---|---|---|---|---:|---:|---|",
            ]
        )
        for row in sorted(summaries, key=lambda item: float(item["median_us"] or -1), reverse=True):
            lines.append(
                f"| `{row['case_id']}` | {row.get('status', 'pass')} | `{row['workload']}` | {row['shape_label']} | "
                f"{row['median_us']} | {row['median_us_per_work_item']} | {priority_bucket(row)} |"
            )
        failed = [row for row in summaries if row.get("status") == "fail"]
        if failed:
            lines.extend(["", "## 失败 Case", ""])
            for row in failed:
                lines.append(f"- `{row['case_id']}`: {row.get('error', '')}")

    lines.extend(
        [
            "",
            "## 解读规则",
            "",
            "- `high` 只表示值得优先做直接 static fork，不表示已经证明 CB FIFO 在 critical path。",
            "- `failed` 表示当前 TTNN baseline path/shape 不可运行，也是一类有效信号：需要先修正真实 op 配置或降低 shape，再做 static fork。",
            "- RMSNorm/Softmax decode 需要继续看 reader/reduce/write 或 reader/SFPU/write 的 device stage breakdown。",
            "- Paged update cache 是最接近 dataflow/write-heavy 的 decode helper，若小 batch latency 高，应优先 fork。",
            "- 下一步对每个 high/medium 候选都要做 `cb`、`static-runtime`、`static-streamreg-cbregs` 或 dataflow-only `static-streamreg-scratch` 的 device profiler 对比。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = build_workload_specs(args)
    write_csv(args.out_dir / "workload_matrix.csv", spec_rows(specs))
    write_json(
        args.out_dir / "manifest.json",
        {
            "repo_root": str(REPO_ROOT),
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "argv": sys.argv,
            "dry_run": args.dry_run,
            "device_id": args.device_id,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "workloads": args.workloads,
            "rmsnorm_layout": args.rmsnorm_layout,
        },
    )

    if args.dry_run:
        render_report(args, specs, [], args.out_dir / "llm_decode_workload_report.md")
        print(f"workload_matrix={args.out_dir / 'workload_matrix.csv'}")
        print(f"report={args.out_dir / 'llm_decode_workload_report.md'}")
        return 0

    for path in (str(REPO_ROOT), str(REPO_ROOT / "ttnn")):
        if path not in sys.path:
            sys.path.insert(0, path)
    pythonpath_parts = [str(REPO_ROOT), str(REPO_ROOT / "ttnn")]
    if os.environ.get("PYTHONPATH"):
        pythonpath_parts.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    import torch
    import ttnn

    torch.manual_seed(args.seed)
    device = ttnn.open_device(device_id=args.device_id)
    raw_rows = []
    summaries = []
    try:
        for spec in specs:
            print(f"running {spec.case_id}", flush=True)
            try:
                case_raw_rows, summary = measure_case(spec, device, torch, ttnn, args)
                raw_rows.extend(case_raw_rows)
                summaries.append(summary)
            except Exception as exc:
                summary = failed_case_summary(spec, args, exc)
                summaries.append(summary)
                print(f"failed {spec.case_id}: {summary['error']}", file=sys.stderr, flush=True)
                if args.stop_on_failure:
                    raise
            write_csv(args.out_dir / "raw_host_times.csv", raw_rows)
            write_csv(args.out_dir / "host_summary.csv", summaries)
    finally:
        ttnn.close_device(device)

    render_report(args, specs, summaries, args.out_dir / "llm_decode_workload_report.md")
    print(f"host_summary={args.out_dir / 'host_summary.csv'}")
    print(f"raw_host_times={args.out_dir / 'raw_host_times.csv'}")
    print(f"report={args.out_dir / 'llm_decode_workload_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
