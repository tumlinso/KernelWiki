#!/usr/bin/env python3
"""Generate deterministic FP16 Q/K/V cases plus the pure-Python FP64 oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from attention_bundle import (InputCase, ReferenceCase, write_input_bundle,
                              write_reference_bundle)
from attention_oracle import Tensor4, attention_fp64, attention_online, write_f16le_bundle


def random_tensor(shape, seed, bound=0.25):
    rng = random.Random(seed)
    return Tensor4.from_values(shape,
                               (rng.uniform(-bound, bound) for _ in range(math.prod(shape))))


def build_case(name: str):
    masked = None
    if name == "prefill64-causal":
        b, hq, hkv, sq, sk, d = 1, 4, 2, 16, 32, 64
        causal, sparse, block = True, False, 0
    elif name == "prefill64-medium-causal":
        b, hq, hkv, sq, sk, d = 1, 4, 2, 64, 128, 64
        causal, sparse, block = True, False, 0
    elif name == "prefill64-long-causal":
        b, hq, hkv, sq, sk, d = 1, 4, 2, 32, 512, 64
        causal, sparse, block = True, False, 0
    elif name == "prefill128-noncausal":
        b, hq, hkv, sq, sk, d = 1, 4, 2, 16, 64, 128
        causal, sparse, block = False, False, 0
    elif name == "decode70-causal":
        b, hq, hkv, sq, sk, d = 1, 8, 2, 1, 4096, 70
        causal, sparse, block = True, False, 0
    elif name == "selected64-causal":
        b, hq, hkv, sq, sk, d = 1, 4, 2, 7, 19, 64
        causal, sparse, block = True, True, 6
    elif name == "score-stress64-noncausal":
        b, hq, hkv, sq, sk, d = 1, 2, 1, 2, 17, 64
        causal, sparse, block = False, False, 0
    else:
        raise ValueError(f"unknown case {name}")
    case_seed = sum(name.encode("utf-8"))
    q = random_tensor((b, hq, sq, d), case_seed + 1)
    k = random_tensor((b, hkv, sk, d), case_seed + 2)
    v = random_tensor((b, hkv, sk, d), case_seed + 3)
    if name == "score-stress64-noncausal":
        q = Tensor4.from_values((b, hq, sq, d), [4.0] * (b * hq * sq * d))
        key_values = []
        stress_rng = random.Random(case_seed + 4)
        for ki in range(sk):
            if ki == 0:
                key_values.extend([4.0] * d)
            elif ki == 1:
                key_values.extend([4.0] * (d - 1) + [3.99609375])
            elif ki == 2:
                key_values.extend([-4.0] * d)
            else:
                key_values.extend(stress_rng.uniform(-4.0, 4.0) for _ in range(d))
        k = Tensor4.from_values((b, hkv, sk, d), key_values)
        v = random_tensor((b, hkv, sk, d), case_seed + 4, 0.25)
    selected = None
    if sparse:
        choices = ([0, 3], [], [1], [3, 0], [2], [1, 3], [3])
        selected = [[[list(choices[qi % len(choices)]) for qi in range(sq)]
                     for _head in range(hq)]]
        masked = [[[[False] * sk for _ in range(sq)] for _ in range(hq)]
                  for _ in range(b)]
        # Selection and an independent per-query mask both constrain row 2.
        masked[0][0][2][6] = True
    return q, k, v, causal, selected, block, masked


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    input_cases = []
    reference_cases = []
    for case_name in ("prefill64-causal", "prefill64-medium-causal",
                      "prefill64-long-causal", "prefill128-noncausal",
                      "decode70-causal", "selected64-causal",
                      "score-stress64-noncausal"):
        q, k, v, causal, selected, block_size, masked = build_case(case_name)
        input_path = args.out / f"{case_name}.f16le"
        meta_path = args.out / f"{case_name}.inputs.json"
        write_f16le_bundle(input_path, q, k, v, meta_path)
        reference_mask = masked
        if selected is not None:
            # The independent semantic reference sees an explicit dense mask
            # reconstructed from caller-supplied IDs, not the sparse-list path.
            reference_mask = [[[[False] * k.shape[2] for _ in range(q.shape[2])]
                               for _ in range(q.shape[1])] for _ in range(q.shape[0])]
            for bi in range(q.shape[0]):
                for hi in range(q.shape[1]):
                    for qi in range(q.shape[2]):
                        ids = set(selected[bi][hi][qi])
                        for ki in range(k.shape[2]):
                            reference_mask[bi][hi][qi][ki] = (
                                ki // block_size not in ids or
                                (masked is not None and masked[bi][hi][qi][ki]))
        ref = attention_fp64(q, k, v, causal=causal,
                             masked=reference_mask)
        online = attention_online(q, k, v,
                                  key_ranges=[(0, 1), (1, max(1, k.shape[2] // 3)),
                                              (max(1, k.shape[2] // 3), k.shape[2])],
                                  causal=causal, selected_blocks=selected,
                                  block_size=block_size or 1, masked=masked)
        delta = max((abs(x - y) for x, y in zip(ref["output"], online["output"])), default=0.0)
        if delta > 2e-11:
            raise SystemExit(f"{case_name}: partitioned CPU merge mismatch {delta}")
        row_max_delta = 0.0
        row_normalizer_delta = 0.0
        row_lse_delta = 0.0
        for ref_row, online_row in zip(ref["rows"], online["rows"]):
            if ref_row["valid_keys"] != online_row["valid_keys"]:
                raise SystemExit(f"{case_name}: selected/masked valid-key mismatch")
            for key in ("max_score", "logsumexp"):
                left, right = ref_row[key], online_row[key]
                if math.isfinite(left) and math.isfinite(right):
                    if key == "max_score":
                        row_max_delta = max(row_max_delta, abs(left - right))
                    else:
                        row_lse_delta = max(row_lse_delta, abs(left - right))
                elif left != right:
                    raise SystemExit(f"{case_name}: mismatched empty-row {key}")
            if ref_row["normalizer"] != online_row["normalizer"]:
                row_normalizer_delta = max(row_normalizer_delta,
                                           abs(ref_row["normalizer"] -
                                               online_row["normalizer"]))
        if max(row_max_delta, row_lse_delta, row_normalizer_delta) > 2e-11:
            raise SystemExit(f"{case_name}: partitioned CPU state mismatch "
                             f"max={row_max_delta}, lse={row_lse_delta}, "
                             f"normalizer={row_normalizer_delta}")
        oracle_path = args.out / f"{case_name}.oracle.json"
        rows_for_json = []
        for row in ref["rows"]:
            rows_for_json.append({key: (value if math.isfinite(value) else None)
                                  for key, value in row.items()})
        oracle_doc = {"format": "KERNELWIKI-CPU-ORACLE/1", "passed": True,
                      "case": case_name,
                      "contract": {"input": "FP16 represented Q/K/V, row-major [B,H,S,D]",
                                   "max_abs_input": {"q": max(abs(x) for x in q.values),
                                                     "k": max(abs(x) for x in k.values),
                                                     "v": max(abs(x) for x in v.values)},
                                   "qk_accumulator": "FP64 semantic oracle; native policy FP32",
                                   "scale": "1/sqrt(D), applied to logits before softmax",
                                   "online_state": "FP32 native (m,l,u), no probability rounding",
                                   "output": "FP32 native; oracle stores FP64 values",
                                   "causal": "right/suffix aligned key position Sk-Sq+query_row"
                                             if causal else "disabled",
                                   "selected_blocks": selected,
                                   "selected_block_size": block_size if selected else None,
                                   "has_per_query_boolean_mask": masked is not None,
                                   "semantic_reference": ("dense boolean mask reconstructed from supplied selected block IDs and combined with the independent mask"
                                                          if selected is not None else
                                                          "dense causal/noncausal mask evaluation"),
                                   "duplicate_block_ids": "rejected",
                                   "all_masked_row": "zero"},
                      "shape": ref["shape"], "rows": rows_for_json,
                      "output_fp64": list(ref["output"]),
                      "online_partition_max_abs_error": delta,
                      "online_partition_max_row_score_error": row_max_delta,
                      "online_partition_max_row_lse_error": row_lse_delta,
                      "online_partition_max_row_normalizer_error": row_normalizer_delta}
        oracle_path.write_text(json.dumps(oracle_doc, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        input_cases.append(InputCase(case_name, q, k, v, causal,
                                     block_size if selected is not None else 0,
                                     selected, masked))
        reference_cases.append(ReferenceCase(case_name, ref["shape"], k.shape,
                                             tuple(ref["output"]), tuple(ref["rows"])))
        cases.append({"id": case_name, "input": input_path.name,
                      "input_sha256": sha256(input_path), "metadata": meta_path.name,
                      "metadata_sha256": sha256(meta_path), "oracle": oracle_path.name,
                      "oracle_sha256": sha256(oracle_path), "shape": ref["shape"],
                      "kv_shape": k.shape, "causal": causal,
                      "head_dim": q.shape[3], "block_size": block_size,
                      "has_per_query_boolean_mask": masked is not None,
                      "selected_blocks": selected,
                      "tolerance_group": ("high-contrast-stress"
                                          if case_name == "score-stress64-noncausal"
                                          else "bounded-default")})
    bundle_path = args.out / "attention_inputs.bin"
    reference_bundle_path = args.out / "attention_reference.bin"
    input_records = write_input_bundle(bundle_path, input_cases)
    reference_records = write_reference_bundle(reference_bundle_path, reference_cases)
    if [record["id"] for record in input_records] != [record["id"] for record in reference_records]:
        raise SystemExit("input/reference case order diverged")
    contract = {"format": "KERNELWIKI-VOLTA-ATTENTION-CONTRACT/1",
                "case_id": "attention",
                "case_signature": {"bundle_format": "KWVATTN1/KWVREF01",
                                   "cases": [{"id": case["id"],
                                              "q_shape": case["shape"],
                                              "kv_shape": case["kv_shape"],
                                              "causal": case["causal"],
                                              "head_dim": case["head_dim"],
                                              "block_size": case["block_size"],
                                              "has_per_query_boolean_mask": case["has_per_query_boolean_mask"],
                                              "tolerance_group": case["tolerance_group"]}
                                             for case in cases]},
                "numerical_contract": {"input": "FP16 represented Q/K/V",
                                       "qk_accumulator": "FP32 native, FP64 semantic oracle",
                                       "scale": "1/sqrt(D) before softmax",
                                       "softmax_state": "FP32 (m,l,u)",
                                       "probability": "FP32, no intermediate rounding",
                                       "output": "FP32",
                                       "output_atol": 1.5e-4,
                                       "output_rtol": 2e-4,
                                       "row_max_atol": 2e-5,
                                       "normalizer_atol": 1e-7,
                                       "normalizer_rtol": 5e-4,
                                       "empty_row": "zero output and zero normalizer",
                                       "causal_alignment": "right aligned; key_position=Sk-Sq+query_row",
                                       "selected_block_duplicates": "rejected",
                                       "frozen_before_native_collection": True,
                                       "tolerance_basis": "fixed before native testing: at D=128 and |Q,K|<=0.25, scaled-score FP32 dot gamma_128 estimate is about 5.4e-6; four-way online partition length is at most 1024 and gamma_1024 is about 6.1e-5 before exp/update rounding; 1.5e-4+2e-4|ref| output and 1e-7+5e-4|l| normalizer thresholds are engineering admission envelopes, not formal bounds; do not enlarge after failure"},
                "bounded_default_domain": {"max_abs_qkv": 0.25,
                                           "max_head_dim": 128,
                                           "max_key_length": 4096},
                "separate_stress_fixtures": [{"case_id": "score-stress64-noncausal",
                                               "purpose": "near-tied leading scores and large finite score contrast",
                                               "max_abs_q_and_k": 4.0,
                                               "inside_bounded_default_domain": False,
                                               "uses_same_fixed_tolerance": True}],
                "represented_input": {"path": bundle_path.name,
                                      "sha256": sha256(bundle_path)},
                "oracle": {"path": reference_bundle_path.name,
                           "sha256": sha256(reference_bundle_path)}}
    (args.out / "attention_contract.json").write_text(
        json.dumps(contract, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "case-index.json").write_text(
        json.dumps({"format": "KERNELWIKI-ATTENTION-CASES/1", "cases": cases,
                    "combined_input": {"path": bundle_path.name,
                                       "sha256": sha256(bundle_path),
                                       "records": input_records},
                    "combined_reference": {"path": reference_bundle_path.name,
                                           "sha256": sha256(reference_bundle_path),
                                           "records": reference_records}},
                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    source_root = Path(__file__).resolve().parents[3]
    source_dir = Path(__file__).resolve().parent
    source_paths = [source_dir / "make_attention_inputs.py",
                    source_dir / "attention_oracle.py",
                    source_dir / "attention_bundle.py",
                    source_root / "artifacts/kernels/volta-attention/variants/volta_attention_runner.cu",
                    source_root / "artifacts/kernels/volta-attention/variants/volta_attention_online.cuh",
                    source_dir / "tests/test_attention_oracle.py",
                    source_dir / "tests/test_attention_bundle.py"]
    source_manifest = {"format": "KERNELWIKI-ATTENTION-SOURCE-MANIFEST/1",
                       "sources": [{"path": str(path.relative_to(source_root)),
                                    "sha256": sha256(path)} for path in source_paths],
                       "artifacts": {"input_bundle": {"path": bundle_path.name,
                                                       "sha256": sha256(bundle_path)},
                                     "reference_bundle": {"path": reference_bundle_path.name,
                                                          "sha256": sha256(reference_bundle_path)},
                                     "contract": {"path": "attention_contract.json",
                                                  "sha256": sha256(args.out / "attention_contract.json")},
                                     "case_index": {"path": "case-index.json",
                                                    "sha256": sha256(args.out / "case-index.json")}},
                       "upstream_code_copied": [],
                       "native_build": "not performed",
                       "native_execution": "not performed",
                       "device_uuid": None,
                       "toolchain": None}
    (args.out / "source-manifest.json").write_text(
        json.dumps(source_manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
