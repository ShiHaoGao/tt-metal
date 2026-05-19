#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
PROFILER_DIR = Path("tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler")


@dataclass(frozen=True)
class FamilySpec:
    name: str
    priority: str
    bottleneck_hypothesis: str
    static_coverage: str
    shape_classes: tuple[str, ...]
    profiler_tasks: tuple[str, ...]
    sweep_modules: tuple[str, ...]
    pytest_tests: tuple[str, ...]
    conclusion_policy: str


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    title: str
    goal: str
    representative_tasks: tuple[str, ...]
    exit_criteria: str
    current_status: str


@dataclass
class TaskRun:
    name: str
    family: str
    phase: str
    kind: str
    command: list[str]
    out_dir: Path
    log_path: Path
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path = REPO_ROOT
    summary_files: tuple[str, ...] = ()
    notes: str = ""
    skip_reason: str = ""


PHASES = (
    PhaseSpec(
        name="phase0",
        title="Evidence baseline",
        goal="Freeze current CB/runtime/static evidence and keep it separate from TTNN family claims.",
        representative_tasks=("cb_protocol_overhead_system",),
        exit_criteria="Protocol overhead, firmware init, and current positive/negative profiler baselines are reproducible.",
        current_status="Complete enough to guide Phase 1-3; rerun when firmware or profiler code changes.",
    ),
    PhaseSpec(
        name="phase1",
        title="Data-movement only, no CB",
        goal="Validate raw L1 staging plus NoC traffic and semaphore synchronization without TT-Metal CB FIFO semantics.",
        representative_tasks=("cb_protocol_overhead_dram", "real_copy"),
        exit_criteria="DRAM traffic is correct and static-runtime / stream-register modes are stable against dram-cb.",
        current_status=(
            "Direct microbenchmark and production-shaped copy fork both run; L1-semaphore copy is negative, "
            "while dataflow-only scratch-register sync nearly matches CB. Compute-path stream-register work "
            "must use per-CB tiles_received/tiles_acked registers."
        ),
    ),
    PhaseSpec(
        name="phase2",
        title="Compiler-managed pseudo-CB",
        goal="Replace CB FIFO state with compiler-owned L1 rings, queue counters, and semaphores.",
        representative_tasks=(
            "static_protocol_modeling_memory_bound",
            "real_tile_add",
            "real_tile_add_multicore",
            "static_protocol_modeling_targeted",
        ),
        exit_criteria="Pseudo-CB keeps overlap, passes correctness, and beats CB on memory-bound pipelines.",
        current_status="Positive for tile-add/eltwise memory-bound paths; matmul protocol model remains limited.",
    ),
    PhaseSpec(
        name="phase3",
        title="TTNN family coverage",
        goal="Classify TTNN operator families by whether static protocol savings survive real TTNN-style paths.",
        representative_tasks=(
            "ttnn_binary_ng_no_bcast",
            "ttnn_binary_ng_no_bcast_multicore",
            "real_matmul_low_k",
            "sweep_*",
            "pytest_*",
        ),
        exit_criteria="Each family has a decision: promote, promote only for specific shapes, or keep CB backend.",
        current_status=(
            "Core coverage complete enough for decisions: eltwise has direct positive TTNN-style data; "
            "matmul is inconclusive; CCL needs a multi-device run; other families have sweep/pytest coverage "
            "and need direct static forks before speedup claims."
        ),
    ),
)


FAMILY_DECISIONS = {
    "protocol_baseline": {
        "phase_state": "phase0 complete, phase1 direct microbenchmark available",
        "current_decision": "Use only as attribution bounds; do not claim TTNN op speedup from microbenchmarks alone.",
        "next_direct_experiment": "Sweep page size, CB count, and RISC count after firmware/runtime changes.",
    },
    "data_movement_layout": {
        "phase_state": "phase1 direct copy fork available; transpose_wh and slice_tile direct forks are weak positive; L1-semaphore copy is negative, stream-register copy is near-CB",
        "current_decision": "Layout/data-movement must be judged op-by-op. transpose_wh and slice_tile have writer-critical weak wins, while copy cannot be promoted.",
        "next_direct_experiment": "Pause before Level C; later add tilize/untilize/concat and multi-core sharded slice with device profiler critical-stage checks.",
    },
    "eltwise": {
        "phase_state": "phase2 positive, phase3 direct positive",
        "current_decision": "Promote memory-bound and simple elementwise paths to compiler-managed L1 experiments.",
        "next_direct_experiment": "Extend no-broadcast add to unary, broadcast, and SFPU-heavy chains.",
    },
    "reduction": {
        "phase_state": "phase3 coverage only",
        "current_decision": "Needs direct fork before promotion; small reductions are the likely first positive case.",
        "next_direct_experiment": "Build sum/mean/max fork with per-stage zones and cross-core reduction shapes.",
    },
    "normalization_softmax": {
        "phase_state": "phase3 coverage only",
        "current_decision": "High-priority LLM target, but decode-like and prefill-like conclusions must stay separate.",
        "next_direct_experiment": "Fork RMSNorm/LayerNorm and softmax decode shapes with reader/reduce/write zones.",
    },
    "matmul_linear": {
        "phase_state": "phase3 direct mixed/inconclusive for current reuse path",
        "current_decision": "Do not promote broad matmul; target low-K, multicast, or decode-like shapes only.",
        "next_direct_experiment": "Add multicast or GEMV-like matmul fork where reader/writer protocol is exposed.",
    },
    "transformer_attention": {
        "phase_state": "phase3 coverage only",
        "current_decision": "Do not use prefill attention as the proof point; prioritize decode helper kernels.",
        "next_direct_experiment": "Profile SDPA decode, rotary, QKV split, and concatenate-heads paths separately.",
    },
    "embedding_kv_cache": {
        "phase_state": (
            "phase3 direct positive for paged_update_cache 8-user decode-like shapes; "
            "embedding lookup is positive; KV cache load-slice/cache-read is near noise; 32-user scalability is still open"
        ),
        "current_decision": (
            "Promote paged_update_cache decode-like update and single-core embedding lookup as shape-specific positive cases. "
            "Do not generalize to all embedding/cache ops; load-slice read is reader-critical and near noise in the current single-core fork."
        ),
        "next_direct_experiment": "Pause before Level C; later fix 32-user/static layout scalability and add multi-core sharded cache-read/padded/chunked embedding forks.",
    },
    "ccl": {
        "phase_state": "phase3 single-card sweep produced no applicable vectors; multi-device run required",
        "current_decision": "Potentially synchronization-bound; local protocol savings must be separated from fabric bottlenecks.",
        "next_direct_experiment": "Add minimal all-gather or reduce-scatter local-ring protocol fork.",
    },
    "conv_pool": {
        "phase_state": "phase3 coverage only",
        "current_decision": "Medium priority control lane; promote only if small/depthwise/pool shapes expose reader/writer cost.",
        "next_direct_experiment": "Run baseline profiling before creating a direct static fork.",
    },
    "creation_fill_typecast": {
        "phase_state": "phase3 coverage only",
        "current_decision": "Useful for host/runtime and writer accounting; keep cold/warm cache split.",
        "next_direct_experiment": "Target fill/typecast small shapes with host enqueue and device zones separated.",
    },
    "backward_moreh_experimental": {
        "phase_state": "phase3 coverage only",
        "current_decision": "Do not aggregate; classify each promoted op by underlying bottleneck.",
        "next_direct_experiment": "Start only after a profiler trace identifies a CB-heavy backward kernel.",
    },
}


FAMILIES = (
    FamilySpec(
        name="protocol_baseline",
        priority="high",
        bottleneck_hypothesis="Isolates CB FIFO, address, semaphore, runtime-arg, and firmware init overhead.",
        static_coverage="direct micro/protocol ablation",
        shape_classes=("small pages", "many CBs", "runtime vs compile-time address"),
        profiler_tasks=("cb_protocol_overhead_system", "cb_protocol_overhead_dram", "static_protocol_modeling_targeted"),
        sweep_modules=(),
        pytest_tests=(),
        conclusion_policy="Use as upper/lower bound only; never generalize directly to a TTNN op.",
    ),
    FamilySpec(
        name="eltwise",
        priority="high",
        bottleneck_hypothesis="Usually memory/protocol exposed, especially no-bcast, bcast, sharded, and short chains.",
        static_coverage="direct TTNN binary-ng fork plus tile-add/chain proxy",
        shape_classes=("small", "medium", "large", "1x1", "2x2"),
        profiler_tasks=("ttnn_binary_ng_no_bcast", "real_tile_add", "static_protocol_modeling_memory_bound"),
        sweep_modules=(
            "eltwise.unary.relu.relu",
            "eltwise.unary.gelu.gelu",
            "eltwise.binary.add.add_all_pytorch2",
            "eltwise.binary.bcast.bcast",
            "eltwise.binary.multiply.multiply",
            "eltwise.ternary.where.where",
        ),
        pytest_tests=(
            "tests/ttnn/unit_tests/operations/eltwise/test_unary.py",
            "tests/ttnn/unit_tests/operations/eltwise/test_add.py",
            "tests/ttnn/unit_tests/operations/eltwise/test_binary_bcast.py",
            "tests/ttnn/unit_tests/operations/eltwise/test_ternary.py",
        ),
        conclusion_policy="Require stable positive delta per local tile and matching host median direction.",
    ),
    FamilySpec(
        name="data_movement_layout",
        priority="high",
        bottleneck_hypothesis="Often DRAM/NoC/protocol bound; layout conversion can hide protocol behind address math.",
        static_coverage=(
            "direct no-CB DRAM microbenchmark plus copy fork; transpose_wh and slice_tile are weak positive; "
            "L1-semaphore copy is negative and dataflow-only scratch-register copy is near-CB, so future "
            "dataflow forks should target tilize, untilize, concat, and multi-core sharded slice; "
            "compute-path streamreg must be per-CB cbregs"
        ),
        shape_classes=("small", "medium", "large", "interleaved", "sharded"),
        profiler_tasks=(
            "cb_protocol_overhead_dram",
            "real_copy",
            "real_copy_multicore",
            "real_tile_add",
            "static_protocol_modeling_memory_bound",
            "ttnn_transpose_wh",
            "ttnn_slice_tile",
        ),
        sweep_modules=(
            "data_movement.copy.copy",
            "data_movement.transpose.transpose_interleaved",
            "data_movement.transpose.transpose_pytorch2",
            "data_movement.slice.slice_pytorch2_tiled",
            "data_movement.concat.concat_interleaved",
            "tilize",
            "untilize",
            "untilize_with_unpadding",
        ),
        pytest_tests=(
            "tests/ttnn/unit_tests/operations/data_movement/test_clone.py",
            "tests/ttnn/unit_tests/operations/data_movement/test_concat.py",
            "tests/ttnn/unit_tests/operations/data_movement/test_slice.py",
            "tests/ttnn/unit_tests/operations/data_movement/test_tilize.py",
            "tests/ttnn/unit_tests/operations/data_movement/test_untilize.py",
        ),
        conclusion_policy="Separate protocol savings from bandwidth ceilings by checking per-byte and per-tile scaling.",
    ),
    FamilySpec(
        name="reduction",
        priority="high",
        bottleneck_hypothesis="Small reductions expose protocol/launch; large reductions move toward pack/unpack, math, or NoC.",
        static_coverage="baseline/proxy today; direct fork needed for sum/mean/max and cross-core reduction",
        shape_classes=("small reduce dim", "long reduce dim", "all-rank", "cross-core"),
        profiler_tasks=("static_protocol_modeling_memory_bound",),
        sweep_modules=(
            "reduction.sum",
            "reduction.mean.mean",
            "reduction.argmax.argmax",
            "reduction.topk.topk",
            "reduction.var.var",
            "reduction.generality.reduction_all_ranks",
        ),
        pytest_tests=(
            "tests/ttnn/unit_tests/operations/reduce/test_sum.py",
            "tests/ttnn/unit_tests/operations/reduce/test_reduction_mean.py",
            "tests/ttnn/unit_tests/operations/reduce/test_argmax.py",
            "tests/ttnn/unit_tests/operations/reduce/test_topk.py",
        ),
        conclusion_policy="Report small-shape protocol win separately from long-reduction bandwidth/compute behavior.",
    ),
    FamilySpec(
        name="normalization_softmax",
        priority="high",
        bottleneck_hypothesis="Decode-like norm/softmax can be latency and memory bound; prefill shapes may become math/pack bound.",
        static_coverage="baseline/proxy today; direct fork should target layernorm/rmsnorm/softmax reader-writer protocol",
        shape_classes=("decode-like", "prefill-like", "sharded", "distributed"),
        profiler_tasks=("static_protocol_modeling_memory_bound",),
        sweep_modules=(
            "normalization.softmax.softmax",
            "normalization.softmax.softmax_sharded",
            "normalization.generality.layernorm",
            "normalization.batch_norm.batch_norm",
            "fused.layer_norm_traces",
            "fused.softmax_traces",
        ),
        pytest_tests=(
            "tests/ttnn/unit_tests/operations/fused/test_layer_norm.py",
            "tests/ttnn/unit_tests/operations/fused/test_rms_norm.py",
            "tests/ttnn/unit_tests/operations/fused/test_group_norm.py",
            "tests/ttnn/unit_tests/operations/fused/test_softmax.py",
        ),
        conclusion_policy="Split decode and prefill conclusions; do not average them into one normalization result.",
    ),
    FamilySpec(
        name="matmul_linear",
        priority="high",
        bottleneck_hypothesis="Large GEMM is usually compute/pack/reuse dominated; low-K and multicast paths can expose protocol.",
        static_coverage="direct real matmul fork exists; multicast/low-K forks are next",
        shape_classes=("low-K", "prefill GEMM", "1D multicast", "2D multicast", "sparse"),
        profiler_tasks=("real_matmul_low_k", "static_protocol_modeling_targeted"),
        sweep_modules=(
            "matmul.short.matmul",
            "matmul.short.matmul_default",
            "matmul.short.matmul_user_program_config_mcast_1d",
            "matmul.short.matmul_user_program_config_mcast_2d",
            "matmul.full.matmul_default_interleaved",
            "matmul.generality.matmul",
            "matmul.generality.linear",
            "matmul.sparse.sparse_matmul",
        ),
        pytest_tests=(
            "tests/ttnn/unit_tests/operations/matmul/test_matmul.py",
            "tests/ttnn/unit_tests/operations/matmul/test_linear.py",
            "tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py",
        ),
        conclusion_policy="Treat real matmul as no stable win until positive deltas survive low-K and multicast schedules.",
    ),
    FamilySpec(
        name="transformer_attention",
        priority="high",
        bottleneck_hypothesis="Decode attention, rotary, QKV split, and head concat are more latency/memory exposed than prefill GEMM.",
        static_coverage="baseline/proxy today; direct forks should target SDPA decode and KV-heavy transformer kernels",
        shape_classes=("decode", "prefill", "paged", "multi-head", "model-traced"),
        profiler_tasks=("real_matmul_low_k", "static_protocol_modeling_memory_bound"),
        sweep_modules=(
            "transformer.attention_softmax.attention_softmax",
            "transformer.rotary_embedding.rotary_embedding",
            "transformer.concatenate_heads.concatenate_heads",
            "transformer.split_query_key_value_and_split_heads.split_query_key_value_and_split_heads",
            "transformer.split_query_key_value_and_split_heads.split_query_key_value_and_split_heads_kv_input",
            "model_traced.attention_softmax__model_traced",
            "model_traced.rotary_embedding_model_traced",
            "model_traced.softmax_model_traced",
        ),
        pytest_tests=(
            "tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py",
            "tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py",
            "tests/ttnn/unit_tests/operations/transformers/test_transformer.py",
            "tests/ttnn/unit_tests/operations/transformers/test_concatenate_heads.py",
        ),
        conclusion_policy="Report decode and prefill separately; decode wins must survive paged-cache traffic.",
    ),
    FamilySpec(
        name="embedding_kv_cache",
        priority="high",
        bottleneck_hypothesis="Lookup/cache update paths are memory and latency exposed; protocol may matter for small token steps.",
        static_coverage=(
            "direct TTNN paged_update_cache update and embedding lookup forks are positive; "
            "KV cache load-slice/cache-read fork is reader-critical and near noise"
        ),
        shape_classes=("decode token", "paged cache", "embedding lookup", "cache fill"),
        profiler_tasks=("ttnn_paged_update_cache", "ttnn_embedding_lookup", "ttnn_kv_cache_load_slice", "static_protocol_modeling_memory_bound"),
        sweep_modules=(
            "embedding.embedding",
            "embedding_bw.embedding_bw",
            "data_movement.embedding.embedding_pytorch2",
            "model_traced.embedding_model_traced",
            "model_traced.fill_cache_model_traced",
            "model_traced.paged_fill_cache_model_traced",
        ),
        pytest_tests=(
            "tests/ttnn/unit_tests/operations/data_movement/test_embedding.py",
            "tests/ttnn/unit_tests/operations/transformers/test_paged_cache_flexible_geometry.py",
            "tests/ttnn/unit_tests/operations/transformers/test_paged_fused_update_cache.py",
        ),
        conclusion_policy="Use token-step latency and per-cache-page cycles, not only aggregate tensor bandwidth.",
    ),
    FamilySpec(
        name="ccl",
        priority="high",
        bottleneck_hypothesis="Collectives are NoC/ethernet/synchronization dominated; local CB savings may or may not survive.",
        static_coverage="baseline/proxy today; direct fork should target all-gather/reduce-scatter minimal kernels",
        shape_classes=("single-chip", "multi-device", "decode", "prefill", "ring"),
        profiler_tasks=("static_protocol_modeling_targeted",),
        sweep_modules=(
            "ccl.generality.all_gather",
            "ccl.generality.all_reduce",
            "ccl.generality.reduce_scatter",
            "ccl.generality.all_broadcast",
            "ccl.generality.point_to_point",
        ),
        pytest_tests=(
            "tests/ttnn/unit_tests/operations/ccl/test_minimals.py",
            "tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py",
            "tests/ttnn/unit_tests/operations/ccl/test_llama_prefill_ccl_ops.py",
        ),
        conclusion_policy="Separate local protocol cost from collective synchronization and fabric bottlenecks.",
    ),
    FamilySpec(
        name="conv_pool",
        priority="medium",
        bottleneck_hypothesis="Conv/pool often bandwidth or compute dominated; small/depthwise/pool cases may expose protocol.",
        static_coverage="baseline/proxy today; direct fork only after baseline shows exposed reader/writer stages",
        shape_classes=("1x1", "depthwise-like", "large conv", "maxpool", "avgpool"),
        profiler_tasks=("static_protocol_modeling_memory_bound",),
        sweep_modules=(
            "conv2d.short.conv2d_short_sweep",
            "conv_transpose2d.short.conv_transpose2d_short_sweep",
            "pool2d.short.max_pool2d_short_sweep",
            "pool2d.short.avg_pool2d_short_sweep",
            "pool2d.global_avg_pool2d",
            "pool2d.max_pool2d",
        ),
        pytest_tests=(
            "tests/ttnn/unit_tests/operations/conv/test_conv2d.py",
            "tests/ttnn/unit_tests/operations/conv/test_conv1d.py",
            "tests/ttnn/unit_tests/operations/pool/test_maxpool2d.py",
            "tests/ttnn/unit_tests/operations/pool/test_avgpool2d.py",
        ),
        conclusion_policy="Use as vision/CNN control lane; staticize only if reader/writer protocol is on critical path.",
    ),
    FamilySpec(
        name="creation_fill_typecast",
        priority="medium",
        bottleneck_hypothesis="Creation/fill/typecast are launch, memory-fill, and conversion heavy; useful for host/runtime accounting.",
        static_coverage="baseline/proxy today; direct fork should target fill/typecast if small shapes show host/runtime dominance",
        shape_classes=("small", "large", "program-cache cold", "program-cache warm"),
        profiler_tasks=("cb_protocol_overhead_system", "static_protocol_modeling_memory_bound"),
        sweep_modules=(
            "creation.empty.empty",
            "creation.zeros.zeros",
            "creation.zeros_like.zeros_like",
            "data_movement.fill.fill_pytorch2",
            "model_traced.fill_model_traced",
            "model_traced.typecast_model_traced",
        ),
        pytest_tests=(
            "tests/ttnn/unit_tests/operations/data_movement/test_creation.py",
            "tests/ttnn/unit_tests/operations/data_movement/test_full.py",
            "tests/ttnn/unit_tests/operations/data_movement/test_full_like.py",
            "tests/ttnn/unit_tests/operations/eltwise/test_eltwise_typecast.py",
        ),
        conclusion_policy="Keep host enqueue, warm-cache, and device cycles separate.",
    ),
    FamilySpec(
        name="backward_moreh_experimental",
        priority="medium",
        bottleneck_hypothesis="Backward and experimental ops mix memory, reduction, matmul, and custom synchronization patterns.",
        static_coverage="baseline only; promote individual kernels to direct forks after CB-heavy profiling",
        shape_classes=("backward", "moreh", "deepseek", "experimental transformer", "ssm"),
        profiler_tasks=("static_protocol_modeling_targeted",),
        sweep_modules=(
            "eltwise.unary_backward.relu_bw.relu_bw",
            "eltwise.binary_backward.add_bw.add_bw",
            "data_movement.backward.concat_bw.concat_bw",
            "reduction.backward.prod_bw.prod_bw",
            "model_traced.moreh_full_model_traced",
            "model_traced.split_query_key_value_and_split_heads_experimental_model_traced",
        ),
        pytest_tests=(
            "tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_matmul.py",
            "tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py",
            "tests/ttnn/unit_tests/operations/deepseek/test_deepseek_prefill_extract.py",
            "tests/ttnn/unit_tests/operations/deepseek/test_deepseek_prefill_insert.py",
        ),
        conclusion_policy="Do not roll up as one number; classify each promoted op by its underlying bottleneck.",
    ),
)


STATIC_SUMMARY_FILES = (
    "device_mode_comparison.csv",
    "host_mode_comparison.csv",
    "critical_stage_summary.csv",
    "combined_device_critical_comparison.csv",
    "combined_host_summary.csv",
    "combined_device_comparison.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run and summarize TTNN static-protocol full-family experiments.")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/ttnn_static_protocol_suite"))
    parser.add_argument("--tier", choices=["coverage", "smoke", "core", "full"], default="coverage")
    parser.add_argument(
        "--arch-name",
        choices=["blackhole", "wormhole_b0"],
        help="TTNN sweep architecture. Defaults to ARCH_NAME/IRD_ARCH_NAME, then ttnn.get_arch_name() when available.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["all", "phase0", "phase1", "phase2", "phase3"],
        default=["all"],
        help="Which experimental phases to include.",
    )
    parser.add_argument("--families", nargs="+", help="Optional family names to include.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--build-dir", type=Path, default=Path("build_Release"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write the plan but do not execute commands.")
    parser.add_argument(
        "--family-sweep-mode",
        choices=["auto", "none", "dry-run", "execute"],
        default="auto",
        help="How to handle sweep-framework coverage for operator families.",
    )
    parser.add_argument(
        "--pytest-mode",
        choices=["auto", "none", "collect-only", "execute"],
        default="auto",
        help="How to handle representative pytest correctness coverage.",
    )
    parser.add_argument(
        "--ttnn-workload-mode",
        choices=["auto", "none", "dry-run", "execute"],
        default="auto",
        help="How to handle real TTNN LLM-decode workload baselines.",
    )
    parser.add_argument("--sweep-suite-name", help="Optional sweep suite name to pass to vector generation/runner.")
    parser.add_argument("--tag", default="ttnn-static-protocol")
    return parser.parse_args()


def selected_families(names):
    if not names:
        return list(FAMILIES)
    known = {family.name: family for family in FAMILIES}
    missing = [name for name in names if name not in known]
    if missing:
        raise RuntimeError(f"Unknown family name(s): {', '.join(missing)}")
    return [known[name] for name in names]


def selected_phases(names):
    if not names or "all" in names:
        return {phase.name for phase in PHASES}
    known = {phase.name for phase in PHASES}
    missing = [name for name in names if name not in known]
    if missing:
        raise RuntimeError(f"Unknown phase name(s): {', '.join(missing)}")
    return set(names)


def command_to_string(command):
    return " ".join(str(part) for part in command)


def resolve_arch_name(explicit_arch_name):
    if explicit_arch_name:
        return explicit_arch_name
    for env_name in ("ARCH_NAME", "IRD_ARCH_NAME"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    try:
        import ttnn

        return str(ttnn.get_arch_name()).strip()
    except Exception:
        return ""


def write_csv_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def list_field(values):
    return ";".join(values)


def family_matrix_rows(families):
    rows = []
    for family in families:
        rows.append(
            {
                "family": family.name,
                "priority": family.priority,
                "bottleneck_hypothesis": family.bottleneck_hypothesis,
                "static_coverage": family.static_coverage,
                "shape_classes": list_field(family.shape_classes),
                "profiler_tasks": list_field(family.profiler_tasks),
                "sweep_modules": list_field(family.sweep_modules),
                "pytest_tests": list_field(family.pytest_tests),
                "conclusion_policy": family.conclusion_policy,
            }
        )
    return rows


def phase_matrix_rows(phases):
    rows = []
    for phase in phases:
        rows.append(
            {
                "phase": phase.name,
                "title": phase.title,
                "goal": phase.goal,
                "representative_tasks": list_field(phase.representative_tasks),
                "exit_criteria": phase.exit_criteria,
                "current_status": phase.current_status,
            }
        )
    return rows


def family_decision_rows(families):
    rows = []
    for family in families:
        decision = FAMILY_DECISIONS.get(family.name, {})
        rows.append(
            {
                "family": family.name,
                "priority": family.priority,
                "phase_state": decision.get("phase_state", ""),
                "current_decision": decision.get("current_decision", ""),
                "next_direct_experiment": decision.get("next_direct_experiment", ""),
            }
        )
    return rows


def validate_family_paths(families):
    rows = []
    for family in families:
        for module in family.sweep_modules:
            path = REPO_ROOT / "tests/sweep_framework/sweeps" / Path(*module.split(".")).with_suffix(".py")
            rows.append(
                {
                    "family": family.name,
                    "kind": "sweep_module",
                    "name": module,
                    "path": str(path.relative_to(REPO_ROOT)),
                    "exists": path.exists(),
                }
            )
        for test in family.pytest_tests:
            path = REPO_ROOT / test
            rows.append(
                {
                    "family": family.name,
                    "kind": "pytest",
                    "name": test,
                    "path": test,
                    "exists": path.exists(),
                }
            )
    return rows


def task_dir(args, name):
    return args.out_dir / "runs" / name


def py_command(script, *extra):
    return [sys.executable, str(REPO_ROOT / script), *map(str, extra)]


def profiler_exe(args, name):
    return REPO_ROOT / args.build_dir / "programming_examples/compiler_managed_l1_dataflow/profiler" / name


def build_task(args):
    targets = [
        "cb_protocol_overhead",
        "real_copy_protocol",
        "real_tile_add_protocol",
        "real_matmul_protocol",
        "static_protocol_modeling",
        "ttnn_binary_ng_no_bcast_protocol",
        "ttnn_binary_ng_sfpu_no_bcast_protocol",
        "ttnn_binary_ng_row_bcast_protocol",
        "ttnn_bcast_to_protocol",
        "ttnn_paged_update_cache_protocol",
        "ttnn_transpose_wh_protocol",
        "ttnn_embedding_lookup_protocol",
        "ttnn_slice_tile_protocol",
        "ttnn_kv_cache_load_slice_protocol",
    ]
    out_dir = task_dir(args, "build_static_protocol_profilers")
    return TaskRun(
        name="build_static_protocol_profilers",
        family="protocol_baseline",
        phase="setup",
        kind="build",
        command=["cmake", "--build", str(REPO_ROOT / args.build_dir), "--target", *targets, "-j", str(args.jobs)],
        out_dir=out_dir,
        log_path=out_dir / "build.log",
        notes="Builds the existing profiler forks used by the suite.",
    )


def static_protocol_tasks(args):
    repeats = args.repeats
    device = args.device_id
    tasks = []

    out_dir = task_dir(args, "cb_protocol_overhead_system")
    tasks.append(
        TaskRun(
            name="cb_protocol_overhead_system",
            family="protocol_baseline",
            phase="phase0",
            kind="static_profiler",
            command=[
                str(profiler_exe(args, "cb_protocol_overhead")),
                "--mode=system",
                "--iterations=1024",
                "--page-size=64",
                "--num-pages=8",
                "--num-cbs=8",
                "--num-rv=2",
                f"--repeats={repeats}",
            ],
            out_dir=out_dir,
            log_path=out_dir / "host.log",
            env={
                "TT_METAL_DEVICE_PROFILER": "1",
                "TT_METAL_CACHE": str(out_dir / "tt_metal_cache"),
            },
            notes="Micro/system baseline for CB FIFO and runtime address protocol.",
        )
    )

    out_dir = task_dir(args, "cb_protocol_overhead_dram")
    tasks.append(
        TaskRun(
            name="cb_protocol_overhead_dram",
            family="data_movement_layout",
            phase="phase1",
            kind="static_profiler",
            command=[
                str(profiler_exe(args, "cb_protocol_overhead")),
                "--mode=dram",
                "--iterations=1024",
                "--page-size=2048",
                "--num-pages=8",
                "--num-cbs=1",
                "--num-rv=2",
                f"--repeats={repeats}",
            ],
            out_dir=out_dir,
            log_path=out_dir / "host.log",
            env={
                "TT_METAL_DEVICE_PROFILER": "1",
                "TT_METAL_CACHE": str(out_dir / "tt_metal_cache"),
            },
            notes="Phase 1 no-CB data-movement check: DRAM read/write through explicit L1 ring plus semaphores.",
        )
    )

    out_dir = task_dir(args, "real_copy")
    copy_tiles = ["256"] if args.tier == "smoke" else ["256", "1024", "4096"]
    tasks.append(
        TaskRun(
            name="real_copy",
            family="data_movement_layout",
            phase="phase1",
            kind="static_profiler",
            command=py_command(
                PROFILER_DIR / "data_movement/real_copy_protocol/run_real_copy_protocol_cases.py",
                "--out-dir",
                out_dir,
                "--tiles",
                *copy_tiles,
                "--num-pages",
                "2",
                "--repeats",
                repeats,
                "--modes",
                "cb",
                "static-runtime",
                "static-compiletime",
                "static-streamreg-scratch",
                "static-streamreg-scratch-compiletime",
                "--device-id",
                device,
            ),
            out_dir=out_dir,
            log_path=out_dir / "suite.log",
            summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
            notes="Production-shaped data-movement-only copy fork with CB, L1-semaphore pseudo-CB, compile-time static, and stream-register static modes.",
        )
    )

    if args.tier in {"core", "full"}:
        out_dir = task_dir(args, "real_copy_multicore")
        tasks.append(
            TaskRun(
                name="real_copy_multicore",
                family="data_movement_layout",
                phase="phase1",
                kind="static_profiler",
                command=py_command(
                    PROFILER_DIR / "data_movement/real_copy_protocol/run_real_copy_protocol_cases.py",
                    "--out-dir",
                    out_dir,
                    "--tiles",
                    "1024",
                    "4096",
                    "--num-pages",
                    "2",
                    "--repeats",
                    repeats,
                    "--modes",
                    "cb",
                    "static-runtime",
                    "static-streamreg-scratch",
                    "--core-grid-x",
                    "2",
                    "--core-grid-y",
                    "2",
                    "--device-id",
                    device,
                ),
                out_dir=out_dir,
                log_path=out_dir / "suite.log",
                summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
                notes="Multi-core data-movement-only protocol penetration check.",
            )
        )

    out_dir = task_dir(args, "static_protocol_modeling_memory_bound")
    tasks.append(
        TaskRun(
            name="static_protocol_modeling_memory_bound",
            family="eltwise",
            phase="phase2",
            kind="static_profiler",
            command=py_command(
                PROFILER_DIR / "compute_pipeline/static_protocol_modeling/run_static_protocol_cases.py",
                "--out-dir",
                out_dir,
                "--case-set",
                "memory-bound",
                "--repeats",
                repeats,
                "--mode",
                "all",
            ),
            out_dir=out_dir,
            log_path=out_dir / "suite.log",
            summary_files=(
                "combined_device_critical_comparison.csv",
                "combined_host_summary.csv",
                "combined_device_comparison.csv",
            ),
            notes="Tile-add and elementwise-chain protocol model.",
        )
    )

    if args.tier in {"core", "full"}:
        out_dir = task_dir(args, "static_protocol_modeling_targeted")
        tasks.append(
            TaskRun(
                name="static_protocol_modeling_targeted",
                family="matmul_linear",
                phase="phase2",
                kind="static_profiler",
                command=py_command(
                    PROFILER_DIR / "compute_pipeline/static_protocol_modeling/run_static_protocol_cases.py",
                    "--out-dir",
                    out_dir,
                    "--case-set",
                    "targeted-device",
                    "--repeats",
                    repeats,
                    "--mode",
                    "all",
                ),
                out_dir=out_dir,
                log_path=out_dir / "suite.log",
                summary_files=(
                    "combined_device_critical_comparison.csv",
                    "combined_host_summary.csv",
                    "combined_device_comparison.csv",
                ),
                notes="Single-core and block matmul protocol model.",
            )
        )

    out_dir = task_dir(args, "real_tile_add")
    tile_args = ["256"] if args.tier == "smoke" else ["256", "1024", "4096"]
    tasks.append(
        TaskRun(
            name="real_tile_add",
            family="eltwise",
            phase="phase2",
            kind="static_profiler",
            command=py_command(
                PROFILER_DIR / "compute_pipeline/real_tile_add_protocol/run_real_tile_add_protocol_cases.py",
                "--out-dir",
                out_dir,
                "--tiles",
                *tile_args,
                "--num-pages",
                "2",
                "--repeats",
                repeats,
                "--modes",
                "cb",
                "static-runtime",
                "static-streamreg-cbregs",
                "--device-id",
                device,
            ),
            out_dir=out_dir,
            log_path=out_dir / "suite.log",
            summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
            notes="Real tile-add reader/compute/writer pipeline.",
        )
    )

    if args.tier in {"core", "full"}:
        out_dir = task_dir(args, "real_tile_add_multicore")
        tasks.append(
            TaskRun(
                name="real_tile_add_multicore",
                family="eltwise",
                phase="phase2",
                kind="static_profiler",
                command=py_command(
                    PROFILER_DIR / "compute_pipeline/real_tile_add_protocol/run_real_tile_add_protocol_cases.py",
                    "--out-dir",
                    out_dir,
                    "--tiles",
                    "1024",
                    "4096",
                    "--num-pages",
                    "2",
                    "--repeats",
                    repeats,
                    "--modes",
                    "cb",
                    "static-runtime",
                    "static-streamreg-cbregs",
                    "--core-grid-x",
                    "2",
                    "--core-grid-y",
                    "2",
                    "--device-id",
                    device,
                ),
                out_dir=out_dir,
                log_path=out_dir / "suite.log",
                summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
                notes="Multi-core memory-bound protocol penetration check.",
            )
        )

    out_dir = task_dir(args, "ttnn_binary_ng_no_bcast")
    binary_tiles = ["1024"] if args.tier == "smoke" else ["1024", "4096", "16384"]
    tasks.append(
        TaskRun(
            name="ttnn_binary_ng_no_bcast",
            family="eltwise",
            phase="phase3",
            kind="static_profiler",
            command=py_command(
                PROFILER_DIR / "ttnn_kernel_forks/ttnn_binary_ng_no_bcast_protocol/run_ttnn_binary_ng_no_bcast_protocol_cases.py",
                "--out-dir",
                out_dir,
                "--tiles",
                *binary_tiles,
                "--num-pages",
                "2",
                "--repeats",
                repeats,
                "--modes",
                "cb",
                "static-runtime",
                "static-streamreg-cbregs",
                "--device-id",
                device,
            ),
            out_dir=out_dir,
            log_path=out_dir / "suite.log",
            summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
            notes="Real TTNN binary-ng no-broadcast add fork.",
        )
    )

    out_dir = task_dir(args, "ttnn_binary_ng_sfpu_no_bcast")
    tasks.append(
        TaskRun(
            name="ttnn_binary_ng_sfpu_no_bcast",
            family="eltwise",
            phase="phase3",
            kind="static_profiler",
            command=py_command(
                PROFILER_DIR
                / "ttnn_kernel_forks/ttnn_binary_ng_sfpu_no_bcast_protocol/run_ttnn_binary_ng_sfpu_no_bcast_protocol_cases.py",
                "--out-dir",
                out_dir,
                "--tiles",
                *binary_tiles,
                "--num-pages",
                "2",
                "--repeats",
                repeats,
                "--modes",
                "cb",
                "static-runtime",
                "static-streamreg-cbregs",
                "--device-id",
                device,
            ),
            out_dir=out_dir,
            log_path=out_dir / "suite.log",
            summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
            notes="Real TTNN binary-ng SFPU no-broadcast div fork; tests whether simple eltwise gains survive compute-heavy SFPU.",
        )
    )

    if args.tier in {"core", "full"}:
        out_dir = task_dir(args, "ttnn_bcast_to_row")
        bcast_tiles = ["1024"] if args.tier == "smoke" else ["1024", "4096", "16384"]
        tasks.append(
            TaskRun(
                name="ttnn_bcast_to_row",
                family="eltwise",
                phase="phase3",
                kind="static_profiler",
                command=py_command(
                    PROFILER_DIR / "ttnn_kernel_forks/ttnn_bcast_to_protocol/run_ttnn_bcast_to_protocol_cases.py",
                    "--out-dir",
                    out_dir,
                    "--tiles",
                    *bcast_tiles,
                    "--width-tiles",
                    "8",
                    "--num-pages",
                    "2",
                    "--repeats",
                    repeats,
                    "--modes",
                    "cb",
                    "static-runtime",
                    "static-streamreg-cbregs",
                    "--device-id",
                    device,
                ),
                out_dir=out_dir,
                log_path=out_dir / "suite.log",
                summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
                notes="Real TTNN experimental bcast_to row-broadcast fork with one input CB and one output CB.",
            )
        )

        out_dir = task_dir(args, "ttnn_binary_ng_row_bcast")
        tasks.append(
            TaskRun(
                name="ttnn_binary_ng_row_bcast",
                family="eltwise",
                phase="phase3",
                kind="static_profiler",
                command=py_command(
                    PROFILER_DIR / "ttnn_kernel_forks/ttnn_binary_ng_row_bcast_protocol/run_ttnn_binary_ng_row_bcast_protocol_cases.py",
                    "--out-dir",
                    out_dir,
                    "--ops",
                    "add",
                    "--tiles",
                    *bcast_tiles,
                    "--width-tiles",
                    "8",
                    "--num-pages",
                    "2",
                    "--repeats",
                    repeats,
                    "--modes",
                    "cb",
                    "static-runtime",
                    "static-streamreg-cbregs",
                    "--device-id",
                    device,
                ),
                out_dir=out_dir,
                log_path=out_dir / "suite.log",
                summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
                notes="Real TTNN binary_ng row-broadcast fork; reader-side software row-bcast add path.",
            )
        )

        out_dir = task_dir(args, "ttnn_binary_ng_no_bcast_multicore")
        tasks.append(
            TaskRun(
                name="ttnn_binary_ng_no_bcast_multicore",
                family="eltwise",
                phase="phase3",
                kind="static_profiler",
                command=py_command(
                    PROFILER_DIR / "ttnn_kernel_forks/ttnn_binary_ng_no_bcast_protocol/run_ttnn_binary_ng_no_bcast_protocol_cases.py",
                    "--out-dir",
                    out_dir,
                    "--tiles",
                    "4096",
                    "16384",
                    "--num-pages",
                    "2",
                    "--repeats",
                    repeats,
                    "--modes",
                    "cb",
                    "static-runtime",
                    "static-streamreg-cbregs",
                    "--core-grid-x",
                    "2",
                    "--core-grid-y",
                    "2",
                    "--device-id",
                    device,
                ),
                out_dir=out_dir,
                log_path=out_dir / "suite.log",
                summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
                notes="Multi-core TTNN binary-ng no-broadcast check.",
            )
        )

        out_dir = task_dir(args, "ttnn_transpose_wh")
        transpose_shapes = ["8x8"] if args.tier == "smoke" else ["8x8", "16x16", "32x32"]
        tasks.append(
            TaskRun(
                name="ttnn_transpose_wh",
                family="data_movement_layout",
                phase="phase3",
                kind="static_profiler",
                command=py_command(
                    PROFILER_DIR / "ttnn_kernel_forks/ttnn_transpose_wh_protocol/run_ttnn_transpose_wh_protocol_cases.py",
                    "--out-dir",
                    out_dir,
                    "--shapes",
                    *transpose_shapes,
                    "--num-pages",
                    "2",
                    "--repeats",
                    repeats,
                    "--modes",
                    "cb",
                    "static-runtime",
                    "static-streamreg-cbregs",
                    "--device-id",
                    device,
                ),
                out_dir=out_dir,
                log_path=out_dir / "suite.log",
                summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
                notes="Real TTNN tiled transpose_wh reader/compute/writer fork; weak layout-family direct evidence.",
            )
        )

        out_dir = task_dir(args, "ttnn_slice_tile")
        slice_shapes = ["64x64x16x64x8x0"] if args.tier == "smoke" else [
            "64x64x16x64x8x0",
            "64x64x32x32x16x16",
            "128x64x64x32x32x16",
        ]
        tasks.append(
            TaskRun(
                name="ttnn_slice_tile",
                family="data_movement_layout",
                phase="phase3",
                kind="static_profiler",
                command=py_command(
                    PROFILER_DIR / "ttnn_kernel_forks/ttnn_slice_tile_protocol/run_ttnn_slice_tile_protocol_cases.py",
                    "--out-dir",
                    out_dir,
                    "--shapes",
                    *slice_shapes,
                    "--num-pages",
                    "2",
                    "--repeats",
                    repeats,
                    "--modes",
                    "cb",
                    "static-runtime",
                    "static-streamreg-cbregs",
                    "--device-id",
                    device,
                ),
                out_dir=out_dir,
                log_path=out_dir / "suite.log",
                summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
                notes="Real TTNN tiled slice reader/writer fork; weak layout-family direct evidence.",
            )
        )

    if args.tier in {"core", "full"}:
        out_dir = task_dir(args, "ttnn_paged_update_cache")
        tasks.append(
            TaskRun(
                name="ttnn_paged_update_cache",
                family="embedding_kv_cache",
                phase="phase3",
                kind="static_profiler",
                command=py_command(
                    PROFILER_DIR
                    / "ttnn_kernel_forks/ttnn_paged_update_cache_protocol/run_ttnn_paged_update_cache_protocol_cases.py",
                    "--out-dir",
                    out_dir,
                    "--users",
                    "8",
                    "--kv-heads",
                    "8",
                    "--head-dims",
                    "128",
                    "--block-sizes",
                    "64",
                    "128",
                    "--max-seq-lens",
                    "2048",
                    "--cache-idxs",
                    "127",
                    "1057",
                    "--num-pages",
                    "2",
                    "--repeats",
                    repeats,
                    "--modes",
                    "cb",
                    "static-runtime",
                    "static-streamreg-cbregs",
                    "--device-id",
                    device,
                ),
                out_dir=out_dir,
                log_path=out_dir / "suite.log",
                summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
                notes=(
                    "Real TTNN paged_update_cache update fork with decode-like 8-user shapes. "
                    "This is Level B static-streamreg-cbregs evidence for embedding/KV-cache update, not for cache read."
                ),
            )
        )

        level_b_out_dir = task_dir(args, "ttnn_paged_update_cache_compiletime_ablation")
        tasks.append(
            TaskRun(
                name="ttnn_paged_update_cache_compiletime_ablation",
                family="embedding_kv_cache",
                phase="phase3",
                kind="static_profiler",
                command=py_command(
                    PROFILER_DIR
                    / "ttnn_kernel_forks/ttnn_paged_update_cache_protocol/run_ttnn_paged_update_cache_protocol_cases.py",
                    "--out-dir",
                    level_b_out_dir,
                    "--users",
                    "1",
                    "--kv-heads",
                    "1",
                    "--head-dims",
                    "128",
                    "--block-sizes",
                    "64",
                    "--max-seq-lens",
                    "2048",
                    "--cache-idxs",
                    "127",
                    "--num-pages",
                    "2",
                    "--repeats",
                    repeats,
                    "--modes",
                    "cb",
                    "static-runtime",
                    "static-streamreg-cbregs",
                    "static-streamreg-cbregs-compiletime",
                    "--device-id",
                    device,
                ),
                out_dir=level_b_out_dir,
                log_path=level_b_out_dir / "suite.log",
                summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
                notes=(
                    "Single-core paged_update_cache compile-time ablation. "
                    "This is not the Level B standard mode; compile-time protocol defines are kernel-global."
                ),
            )
        )

        out_dir = task_dir(args, "ttnn_embedding_lookup")
        embedding_shapes = ["64x1024x128"] if args.tier == "smoke" else ["256x32000x128", "1024x32000x128", "4096x32000x128"]
        tasks.append(
            TaskRun(
                name="ttnn_embedding_lookup",
                family="embedding_kv_cache",
                phase="phase3",
                kind="static_profiler",
                command=py_command(
                    PROFILER_DIR
                    / "ttnn_kernel_forks/ttnn_embedding_lookup_protocol/run_ttnn_embedding_lookup_protocol_cases.py",
                    "--out-dir",
                    out_dir,
                    "--shapes",
                    *embedding_shapes,
                    "--num-pages",
                    "2",
                    "--repeats",
                    repeats,
                    "--modes",
                    "cb",
                    "static-runtime",
                    "static-streamreg-cbregs",
                    "--device-id",
                    device,
                ),
                out_dir=out_dir,
                log_path=out_dir / "suite.log",
                summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
                notes="Real TTNN-style embedding lookup/read-heavy fork; first direct evidence for embedding lookup.",
            )
        )

        out_dir = task_dir(args, "ttnn_kv_cache_load_slice")
        kv_load_shapes = ["128x32x4x0"] if args.tier == "smoke" else [
            "128x32x4x0",
            "512x64x4x64",
            "1024x128x4x128",
        ]
        tasks.append(
            TaskRun(
                name="ttnn_kv_cache_load_slice",
                family="embedding_kv_cache",
                phase="phase3",
                kind="static_profiler",
                command=py_command(
                    PROFILER_DIR
                    / "ttnn_kernel_forks/ttnn_kv_cache_load_slice_protocol/run_ttnn_kv_cache_load_slice_protocol_cases.py",
                    "--out-dir",
                    out_dir,
                    "--shapes",
                    *kv_load_shapes,
                    "--repeats",
                    repeats,
                    "--modes",
                    "cb",
                    "static-runtime",
                    "static-streamreg-cbregs",
                    "--device-id",
                    device,
                ),
                out_dir=out_dir,
                log_path=out_dir / "suite.log",
                summary_files=("device_mode_comparison.csv", "host_mode_comparison.csv", "critical_stage_summary.csv"),
                notes=(
                    "Real TTNN nlp_kv_cache_load_slice-style cache-read fork. "
                    "Current single-core shapes are kept within L1 capacity and are expected to be reader-critical."
                ),
            )
        )

    out_dir = task_dir(args, "real_matmul_low_k")
    matmul_ks = ["64"] if args.tier == "smoke" else ["64", "128", "256"]
    tasks.append(
        TaskRun(
            name="real_matmul_low_k",
            family="matmul_linear",
            phase="phase3",
            kind="static_profiler",
            command=py_command(
                PROFILER_DIR / "ttnn_kernel_forks/real_matmul_protocol/run_real_matmul_protocol_cases.py",
                "--out-dir",
                out_dir,
                "--dims",
                "512",
                "--Ks",
                *matmul_ks,
                "--num-pages",
                "2",
                "--repeats",
                repeats,
                "--modes",
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
                "--device-id",
                device,
            ),
            out_dir=out_dir,
            log_path=out_dir / "suite.log",
            summary_files=("device_mode_comparison.csv", "critical_stage_summary.csv"),
            notes="Real matmul reuse path, low-K protocol exposure check.",
        )
    )

    if args.tier == "full":
        out_dir = task_dir(args, "real_matmul_large_decode_prefill_mix")
        tasks.append(
            TaskRun(
                name="real_matmul_large_decode_prefill_mix",
                family="matmul_linear",
                phase="phase3",
                kind="static_profiler",
                command=py_command(
                    PROFILER_DIR / "ttnn_kernel_forks/real_matmul_protocol/run_real_matmul_protocol_cases.py",
                    "--out-dir",
                    out_dir,
                    "--dims",
                    "512",
                    "1024",
                    "--Ks",
                    "64",
                    "128",
                    "256",
                    "--num-pages",
                    "2",
                    "--repeats",
                    repeats,
                    "--modes",
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
                    "--skip-check",
                    "--device-id",
                    device,
                ),
                out_dir=out_dir,
                log_path=out_dir / "suite.log",
                summary_files=("device_mode_comparison.csv", "critical_stage_summary.csv"),
                notes="Wider matmul sweep; correctness can be skipped for known long cases.",
            )
        )

    return tasks


def sweep_mode(args):
    if args.family_sweep_mode != "auto":
        return args.family_sweep_mode
    if args.tier in {"coverage", "smoke"}:
        return "none"
    if args.tier == "core":
        return "dry-run"
    return "execute"


def pytest_mode(args):
    if args.pytest_mode != "auto":
        return args.pytest_mode
    if args.tier in {"coverage", "smoke"}:
        return "none"
    if args.tier == "core":
        return "collect-only"
    return "execute"


def ttnn_workload_mode(args):
    if args.ttnn_workload_mode != "auto":
        return args.ttnn_workload_mode
    if args.tier in {"coverage", "smoke"}:
        return "dry-run"
    return "execute"


def modules_for_family(family, tier):
    if tier == "smoke":
        return family.sweep_modules[:1]
    if tier == "core":
        return family.sweep_modules[: min(3, len(family.sweep_modules))]
    return family.sweep_modules


def tests_for_family(family, tier):
    if tier == "smoke":
        return family.pytest_tests[:1]
    if tier == "core":
        return family.pytest_tests[: min(2, len(family.pytest_tests))]
    return family.pytest_tests


def family_sweep_tasks(args, families):
    mode = sweep_mode(args)
    if mode == "none":
        return []
    tasks = []
    for family in families:
        modules = modules_for_family(family, args.tier)
        if not modules:
            continue

        suite_args = ["--suite-name", args.sweep_suite_name] if args.sweep_suite_name else []
        for module in modules:
            module_key = module.replace(".", "_")
            module_out_dir = task_dir(args, f"sweep_{family.name}_{module_key}")
            tasks.append(
                TaskRun(
                    name=f"sweep_generate_{family.name}_{module}",
                    family=family.name,
                    phase="phase3",
                    kind="sweep_vector_generation",
                    command=py_command(
                        "tests/sweep_framework/sweeps_parameter_generator.py",
                        "--module-name",
                        module,
                        *suite_args,
                        "--tag",
                        args.tag,
                    ),
                    out_dir=module_out_dir,
                    log_path=module_out_dir / "vector_generation.log",
                    notes="Generates sweep vectors for one representative module.",
                )
            )

            runner_args = [
                "--module-name",
                module,
                "--vector-source",
                "vectors_export",
                "--result-dest",
                "results_export",
                "--tag",
                args.tag,
                "--summary",
            ]
            if args.sweep_suite_name:
                runner_args.extend(["--suite-name", args.sweep_suite_name])
            if mode == "dry-run":
                runner_args.append("--dry-run")
            if mode == "execute":
                runner_args.extend(["--perf", "--device-perf"])
            tasks.append(
                TaskRun(
                    name=f"sweep_run_{family.name}_{module}",
                    family=family.name,
                    phase="phase3",
                    kind=f"sweep_{mode}",
                    command=py_command("tests/sweep_framework/sweeps_runner.py", *runner_args),
                    out_dir=module_out_dir,
                    log_path=module_out_dir / "sweeps_runner.log",
                    env={"TT_METAL_DEVICE_PROFILER": "1"} if mode == "execute" else {},
                    notes="Runs or dry-runs one representative sweep module.",
                )
            )
    return tasks


def ttnn_workload_tasks(args, families):
    mode = ttnn_workload_mode(args)
    if mode == "none":
        return []

    family_names = {family.name for family in families}
    tasks = []
    dry_run_arg = ["--dry-run"] if mode == "dry-run" else []
    common_args = [
        "--device-id",
        args.device_id,
        "--repeats",
        args.repeats,
    ]

    if "normalization_softmax" in family_names:
        out_dir = task_dir(args, "ttnn_llm_decode_norm_softmax")
        command = py_command(
            PROFILER_DIR / "ttnn_workloads/llm_decode/run_ttnn_llm_decode_workloads.py",
            "--out-dir",
            out_dir,
            "--workloads",
            "rmsnorm",
            "softmax-decode",
            "--rmsnorm-seq-lens",
            "1",
            *([] if args.tier == "smoke" else ["16"]),
            "--rmsnorm-hidden-dims",
            "8192",
            "--softmax-heads",
            "64",
            "--softmax-kv-tokens",
            "128",
            "1024",
            *([] if args.tier == "smoke" else ["4096", "8192"]),
            *common_args,
            *dry_run_arg,
        )
        tasks.append(
            TaskRun(
                name="ttnn_llm_decode_norm_softmax",
                family="normalization_softmax",
                phase="phase3",
                kind="ttnn_workload_baseline",
                command=command,
                out_dir=out_dir,
                log_path=out_dir / "workload.log",
                summary_files=("host_summary.csv", "workload_matrix.csv", "llm_decode_workload_report.md"),
                notes=(
                    "Real TTNN RMSNorm and decode softmax host end-to-end workload baseline; "
                    "use this to prioritize direct static-protocol forks."
                ),
            )
        )

    if "embedding_kv_cache" in family_names:
        out_dir = task_dir(args, "ttnn_llm_decode_paged_update_cache")
        command = py_command(
            PROFILER_DIR / "ttnn_workloads/llm_decode/run_ttnn_llm_decode_workloads.py",
            "--out-dir",
            out_dir,
            "--workloads",
            "paged-update-cache",
            "--paged-users",
            "1",
            *([] if args.tier == "smoke" else ["16"]),
            "--paged-kv-heads",
            "8",
            "--paged-head-dims",
            "128",
            "--paged-block-sizes",
            "32",
            "--paged-max-seq-lens",
            "2048",
            *common_args,
            *dry_run_arg,
        )
        tasks.append(
            TaskRun(
                name="ttnn_llm_decode_paged_update_cache",
                family="embedding_kv_cache",
                phase="phase3",
                kind="ttnn_workload_baseline",
                command=command,
                out_dir=out_dir,
                log_path=out_dir / "workload.log",
                summary_files=("host_summary.csv", "workload_matrix.csv", "llm_decode_workload_report.md"),
                notes=(
                    "Real TTNN paged_update_cache decode-step host end-to-end workload baseline; "
                    "paged cache is the next high-priority direct fork target if latency is exposed."
                ),
            )
        )
    return tasks


def pytest_tasks(args, families):
    mode = pytest_mode(args)
    if mode == "none":
        return []
    tasks = []
    for family in families:
        tests = tests_for_family(family, args.tier)
        if not tests:
            continue
        out_dir = task_dir(args, f"pytest_{family.name}")
        command = [sys.executable, "-m", "pytest", "-q", *tests]
        if mode == "collect-only":
            command.append("--collect-only")
        skip_reason = ""
        if family.name == "ccl" and mode == "collect-only":
            skip_reason = "CCL pytest collection imports multi-device/demo setup and can open devices on single-card systems."
        tasks.append(
            TaskRun(
                name=f"pytest_{family.name}",
                family=family.name,
                phase="phase3",
                kind=f"pytest_{mode}",
                command=command,
                out_dir=out_dir,
                log_path=out_dir / "pytest.log",
                notes="Representative correctness coverage for this family.",
                skip_reason=skip_reason,
            )
        )
    return tasks


def build_task_plan(args, families, phases):
    tasks = []
    family_names = {family.name for family in families}
    if args.tier != "coverage":
        tasks.extend(
            task
            for task in static_protocol_tasks(args)
            if task.phase in phases and (not args.families or task.family in family_names)
        )
    if "phase3" in phases:
        tasks.extend(ttnn_workload_tasks(args, families))
        tasks.extend(family_sweep_tasks(args, families))
        tasks.extend(pytest_tasks(args, families))
    if any(task.kind == "static_profiler" for task in tasks) and args.tier != "coverage" and not args.skip_build:
        tasks.insert(0, build_task(args))
    return tasks


def prepare_task(task):
    task.out_dir.mkdir(parents=True, exist_ok=True)
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    (task.out_dir / "command.txt").write_text(command_to_string(task.command) + "\n", encoding="utf-8")
    if task.notes:
        (task.out_dir / "notes.txt").write_text(task.notes + "\n", encoding="utf-8")


def build_task_env(task):
    env = os.environ.copy()
    env.update(task.env)
    env.setdefault("TT_METAL_HOME", str(REPO_ROOT))
    if task.kind.startswith("sweep_"):
        env.setdefault("MESH_DEVICE_SHAPE", "1x1")
        if not env.get("TEST_GROUP_NAME"):
            if env.get("ARCH_NAME") == "blackhole":
                env["TEST_GROUP_NAME"] = "blackhole-p150b-sweeps"
            else:
                env["TEST_GROUP_NAME"] = "wormhole-n150-sweeps"
    pythonpath_parts = [
        str(REPO_ROOT),
        str(REPO_ROOT / "ttnn"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def run_task(task, dry_run):
    prepare_task(task)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    row = {
        "name": task.name,
        "family": task.family,
        "phase": task.phase,
        "kind": task.kind,
        "status": "planned" if dry_run else "running",
        "returncode": "",
        "duration_s": "",
        "started_utc": started,
        "log_path": str(task.log_path),
        "out_dir": str(task.out_dir),
        "command": command_to_string(task.command),
        "notes": task.notes,
    }
    if task.skip_reason:
        row["notes"] = f"{task.notes} {task.skip_reason}".strip()
    if dry_run:
        row["status"] = "planned"
        return row
    if task.skip_reason:
        row["status"] = "skip"
        row["returncode"] = "0"
        row["duration_s"] = "0.000"
        with task.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"skipped: {task.skip_reason}\n")
        return row

    env = build_task_env(task)
    start = time.monotonic()
    with task.log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {command_to_string(task.command)}\n")
        log_file.flush()
        process = subprocess.run(task.command, cwd=task.cwd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    duration = time.monotonic() - start
    row["returncode"] = process.returncode
    row["duration_s"] = f"{duration:.3f}"
    if process.returncode == 0:
        row["status"] = "pass"
    elif is_empty_sweep_vector_failure(task):
        row["status"] = "skip"
        row["notes"] = f"{task.notes} No applicable sweep vectors were generated for this local hardware/configuration."
    elif is_pytest_collect_import_failure(task):
        row["status"] = "skip"
        row["notes"] = f"{task.notes} Pytest collect-only could not import all representative tests in this environment."
    else:
        row["status"] = "fail"
    if process.returncode == 0 and task.name.startswith("cb_protocol_overhead_"):
        postprocess_cb_protocol(task)
    return row


def is_empty_sweep_vector_failure(task):
    if not task.kind.startswith("sweep_") or not task.log_path.exists():
        return False
    text = task.log_path.read_text(encoding="utf-8", errors="replace")
    return "must contain a non-empty 'vector_files' list" in text


def is_pytest_collect_import_failure(task):
    if task.kind != "pytest_collect-only" or not task.log_path.exists():
        return False
    text = task.log_path.read_text(encoding="utf-8", errors="replace")
    return "ERROR collecting" in text and "ModuleNotFoundError" in text


def cb_protocol_iterations(task):
    for index, part in enumerate(task.command):
        part = str(part)
        if part.startswith("--iterations="):
            return part.split("=", 1)[1]
        if part == "--iterations" and index + 1 < len(task.command):
            return str(task.command[index + 1])
    return "1"


def postprocess_cb_protocol(task):
    generated = REPO_ROOT / "generated/profiler/.logs/profile_log_device.csv"
    if not generated.exists():
        return
    device_csv = task.out_dir / "profile_log_device.csv"
    shutil.copy2(generated, device_csv)
    analysis_log = task.out_dir / "analysis.txt"
    command = py_command(
        PROFILER_DIR / "microbench/cb_protocol_overhead/analyze_profile_csv.py",
        device_csv,
        "--iterations",
        cb_protocol_iterations(task),
    )
    with analysis_log.open("w", encoding="utf-8") as log_file:
        subprocess.run(command, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, check=False)


def collect_static_summary_rows(tasks):
    rows = []
    for task in tasks:
        if task.kind != "static_profiler":
            continue
        summary_files = task.summary_files or STATIC_SUMMARY_FILES
        for relative in summary_files:
            path = task.out_dir / relative
            if not path.exists() or path.stat().st_size == 0:
                continue
            with path.open(newline="", encoding="utf-8") as csv_file:
                for row in csv.DictReader(csv_file):
                    row = dict(row)
                    row["suite_task"] = task.name
                    row["suite_family"] = task.family
                    row["suite_phase"] = task.phase
                    row["source_csv"] = str(path)
                    rows.append(row)
    return rows


def collect_ttnn_workload_rows(tasks):
    rows = []
    for task in tasks:
        if task.kind != "ttnn_workload_baseline":
            continue
        summary_path = task.out_dir / "host_summary.csv"
        if not summary_path.exists() or summary_path.stat().st_size == 0:
            continue
        with summary_path.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                row = dict(row)
                row["suite_task"] = task.name
                row["suite_family"] = task.family
                row["suite_phase"] = task.phase
                row["source_csv"] = str(summary_path)
                rows.append(row)
    return rows


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def task_plan_rows(tasks):
    return [
        {
            "name": task.name,
            "family": task.family,
            "phase": task.phase,
            "kind": task.kind,
            "out_dir": str(task.out_dir),
            "log_path": str(task.log_path),
            "command": command_to_string(task.command),
            "notes": task.notes,
            "skip_reason": task.skip_reason,
        }
        for task in tasks
    ]


def render_markdown(args, phase_rows, families, decision_rows, tasks, task_results, path_validation, path):
    executed = [row for row in task_results if row["status"] == "pass"]
    failed = [row for row in task_results if row["status"] == "fail"]
    skipped = [row for row in task_results if row["status"] == "skip"]
    planned = [row for row in task_results if row["status"] == "planned"]
    missing_paths = [row for row in path_validation if not row["exists"]]

    lines = [
        "# TTNN Static Protocol Operator-Family Suite",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"Tier: `{args.tier}`",
        f"Phases: `{','.join(args.phases)}`",
        f"Out dir: `{args.out_dir}`",
        "",
        "## 状态",
        "",
        f"- Passed tasks: {len(executed)}",
        f"- Failed tasks: {len(failed)}",
        f"- Skipped tasks: {len(skipped)}",
        f"- Planned-only tasks: {len(planned)}",
        f"- Missing referenced paths: {len(missing_paths)}",
        "",
        "## 实验阶段",
        "",
        "| Phase | Goal | Status |",
        "|---|---|---|",
    ]
    for row in phase_rows:
        lines.append(f"| `{row['phase']}` {row['title']} | {row['goal']} | {row['current_status']} |")

    lines.extend(
        [
            "",
            "## 算子 Family",
            "",
            "| Family | Priority | Static coverage | Bottleneck hypothesis |",
            "|---|---|---|---|",
        ]
    )
    for family in families:
        lines.append(
            f"| `{family.name}` | {family.priority} | {family.static_coverage} | {family.bottleneck_hypothesis} |"
        )

    lines.extend(
        [
            "",
            "## Family 决策",
            "",
            "| Family | Phase state | Current decision | Next direct experiment |",
            "|---|---|---|---|",
        ]
    )
    for row in decision_rows:
        lines.append(
            f"| `{row['family']}` | {row['phase_state']} | {row['current_decision']} | {row['next_direct_experiment']} |"
        )

    lines.extend(
        [
            "",
            "## 任务计划",
            "",
            "| Task | Phase | Family | Kind | Status | Log |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in task_results:
        lines.append(
            f"| `{row['name']}` | `{row['phase']}` | `{row['family']}` | `{row['kind']}` | {row['status']} | `{row['log_path']}` |"
        )

    if missing_paths:
        lines.extend(["", "## 缺失引用", ""])
        for row in missing_paths:
            lines.append(f"- `{row['kind']}` `{row['name']}` -> `{row['path']}`")

    lines.extend(
        [
            "",
            "## 解读规则",
            "",
            "- decode-like 和 prefill-like transformer 行为必须分开报告。",
            "- TTNN workload baseline 使用 host end-to-end timing，只用于选择下一批 direct fork，不是 static-protocol speedup 证据。",
            "- direct static-profiler 的 device delta 比 baseline sweep timing 更强。",
            "- 在称为真实 kernel win 前，device critical-path savings 必须随 local work items 稳定缩放。",
            "- host enqueue/finish、cold/warm cache、device critical path 必须分开下结论。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    arch_name = resolve_arch_name(args.arch_name)
    if arch_name:
        os.environ.setdefault("ARCH_NAME", arch_name)
        os.environ.setdefault("IRD_ARCH_NAME", arch_name)
    os.environ.setdefault("TT_METAL_HOME", str(REPO_ROOT))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    phases = selected_phases(args.phases)
    selected_phase_specs = [phase for phase in PHASES if phase.name in phases]
    families = selected_families(args.families)

    phase_rows = phase_matrix_rows(selected_phase_specs)
    family_rows = family_matrix_rows(families)
    decision_rows = family_decision_rows(families)
    path_validation = validate_family_paths(families)
    tasks = build_task_plan(args, families, phases)

    write_csv_rows(args.out_dir / "phase_matrix.csv", phase_rows)
    write_csv_rows(args.out_dir / "operator_family_matrix.csv", family_rows)
    write_csv_rows(args.out_dir / "family_decision_matrix.csv", decision_rows)
    write_csv_rows(args.out_dir / "path_validation.csv", path_validation)
    write_csv_rows(args.out_dir / "task_plan.csv", task_plan_rows(tasks))
    write_json(
        args.out_dir / "suite_manifest.json",
        {
            "tier": args.tier,
            "phases": sorted(phases),
            "repeats": args.repeats,
            "arch_name": arch_name,
            "phase_matrix": phase_rows,
            "families": family_rows,
            "family_decisions": decision_rows,
            "tasks": task_plan_rows(tasks),
            "path_validation": path_validation,
        },
    )

    task_results = []
    for task in tasks:
        print(f"{'planning' if args.dry_run else 'running'} {task.name}", flush=True)
        result = run_task(task, args.dry_run)
        task_results.append(result)
        if result["status"] == "fail":
            break

    write_csv_rows(args.out_dir / "task_results.csv", task_results)
    static_rows = collect_static_summary_rows(tasks)
    write_csv_rows(args.out_dir / "static_protocol_summary.csv", static_rows)
    ttnn_workload_rows = collect_ttnn_workload_rows(tasks)
    write_csv_rows(args.out_dir / "ttnn_workload_summary.csv", ttnn_workload_rows)
    render_markdown(
        args,
        phase_rows,
        families,
        decision_rows,
        tasks,
        task_results,
        path_validation,
        args.out_dir / "suite_summary.md",
    )

    failed = [row for row in task_results if row["status"] == "fail"]
    if failed:
        print(f"failed task: {failed[0]['name']}; see {failed[0]['log_path']}", file=sys.stderr)
        return 1

    print(f"suite_summary={args.out_dir / 'suite_summary.md'}")
    print(f"phase_matrix={args.out_dir / 'phase_matrix.csv'}")
    print(f"operator_family_matrix={args.out_dir / 'operator_family_matrix.csv'}")
    print(f"family_decision_matrix={args.out_dir / 'family_decision_matrix.csv'}")
    print(f"task_plan={args.out_dir / 'task_plan.csv'}")
    if static_rows:
        print(f"static_protocol_summary={args.out_dir / 'static_protocol_summary.csv'}")
    if ttnn_workload_rows:
        print(f"ttnn_workload_summary={args.out_dir / 'ttnn_workload_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
