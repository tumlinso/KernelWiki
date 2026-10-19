import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).parents[1] / "scripts" / "volta_evidence_check.py"
SPEC = importlib.util.spec_from_file_location("volta_evidence_check", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
SHIM = Path(__file__).parents[1] / "benchmark_runs" / "volta-environment" / "ncu-collection-shim.py"


def put(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def json_file(path: Path, data) -> str:
    return put(path, (json.dumps(data, sort_keys=True) + "\n").encode())


class VoltaEvidenceCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        self.base = self.root / "benchmark_runs" / "volta-case"
        (self.root / ".git").mkdir(parents=True)
        (self.root / "data").mkdir()
        (self.root / "data" / "schemas.yaml").write_text("test\n")
        self.repo_shim = self.root / "benchmark_runs" / "volta-environment" / "ncu-collection-shim.py"
        self.repo_shim_sha = put(self.repo_shim, SHIM.read_bytes())
        self.fake_ncu = self.root / "fake-ncu"
        self.fake_ncu_sha = put(self.fake_ncu, b"test fake Nsight Compute executable")
        self.contract = self.base / "contracts" / "case.json"
        self.contract_sha = json_file(self.contract, {
            "case_id": "m16n16k16",
            "case_signature": {"m": 16, "n": 16, "k": 16, "layout": "row_major"},
            "numerical_contract": {"input": "fp16", "accumulator": "fp32", "output": "fp32", "atol": 0.002, "rtol": 0.002},
        })
        self.input_path = self.root / "benchmark_runs" / "volta-case" / "inputs" / "represented.bin"
        self.input_sha = put(self.input_path, b"represented fp16 input bytes")
        self.impl = {}
        for name in ("baseline", "candidate"):
            source = self.root / "artifacts" / "kernels" / "example" / (name + ".cu")
            binary = self.root / "benchmark_runs" / "volta-case" / "bin" / name
            self.impl[name] = {
                "source": {"path": str(source.relative_to(self.root)), "sha256": put(source, (name + " source\n").encode())},
                "binary": {"path": str(binary.relative_to(self.root)), "sha256": put(binary, (name + " binary\n").encode())},
                "source_path": source,
                "binary_path": binary,
            }
        self.oracle_path = self.base / "oracle" / "case.json"
        self.oracle_sha = json_file(self.oracle_path, {
            "format": "KERNELWIKI-CPU-ORACLE/1", "passed": True,
            "checks": [{"max_abs_error": 0.0001, "reference_max_abs": 1.0, "atol": 0.002, "rtol": 0.002}],
        })
        self.gpu = "GPU-00000000-0000-0000-0000-000000000001"
        self.runs = {}
        self.ids = 0
        for role, (kind, implementation) in CHECK.REQUIRED_RUNS.items():
            self.runs[role] = self.make_run(role, kind, self.impl[implementation], timing=role.endswith("timing"))
        self.sanitizer_summary = self.base / "profiles" / "compute-sanitizer-summary.json"
        self.sanitizer_sha = json_file(self.sanitizer_summary, {"status": "passed", "error_count": 0})
        report = self.base / "profiles" / "selected.ncu-rep"
        report_sha = put(report, b"NCU report bytes selected kernel\n")
        analysis = self.base / "profiles" / "analysis.json"
        analysis_sha = json_file(analysis, {"selected_kernel": "wmma_16x16x16", "readable": True, "metrics": {"sm__throughput": 0.42}})
        shim_log = self.base / "profiles" / "ncu-shim.jsonl"
        shim_record = {
            "policy": "collection-clock-control-none-source-import-yes",
            "shim_path": str(self.repo_shim), "shim_sha256": self.repo_shim_sha,
            "actual_ncu_path": str(self.fake_ncu), "actual_ncu_sha256": self.fake_ncu_sha,
            "argv_received": ["--set", "full", "--launch-count", "1"],
            "argv_executed": ["--import-source", "yes", "--clock-control", "none",
                              "--set", "full", "--launch-count", "1"],
        }
        shim_sha = put(shim_log, (json.dumps(shim_record, sort_keys=True) + "\n").encode())
        self.runs["profile_candidate"]["profile"] = {
            "selected_kernel": "wmma_16x16x16",
            "report": {"path": str(report.relative_to(self.base)), "sha256": report_sha},
            "analysis": {"path": str(analysis.relative_to(self.base)), "sha256": analysis_sha},
            "shim_log": {"path": str(shim_log.relative_to(self.base)), "sha256": shim_sha},
        }
        self.runs["sanitizer_candidate"]["sanitizer_summary"] = {
            "path": str(self.sanitizer_summary.relative_to(self.base)), "sha256": self.sanitizer_sha,
        }
        self.manifest_path = self.base / "summary.json"
        self.manifest = {
            "format": CHECK.FORMAT,
            "cases": [{
                "id": "m16n16k16", "disposition": "evaluated_not_promoted", "gpu_uuid": self.gpu,
                "libraries": [],
                "contract": {"path": "contracts/case.json", "sha256": self.contract_sha},
                "represented_input": {"path": str(self.input_path.relative_to(self.root)), "sha256": self.input_sha},
                "implementations": {name: {"source": entry["source"], "binary": entry["binary"]} for name, entry in self.impl.items()},
                "oracle": {"path": str(self.oracle_path.relative_to(self.base)), "sha256": self.oracle_sha},
                "runs": self.runs,
            }],
        }
        json_file(self.manifest_path, self.manifest)

    def tearDown(self):
        self.temp.cleanup()

    def reset_fixture(self):
        self.temp.cleanup()
        self.setUp()

    def make_run(self, role, kind, implementation, timing=False):
        self.ids += 1
        evidence_id = f"evidence-{self.ids}"
        run_dir = self.base / "controller" / role
        lease_path = self.root / ".todo-orchestrator" / "runtime" / "foreground" / ("run-" + role) / "lease.json"
        lease_sha = json_file(lease_path, {
            "format": "CUDA-FOREGROUND-LEASE/1", "project_root": str(self.root),
            "resource_ids": ["accelerator:" + self.gpu], "owner_id": "owner-" + role,
        })
        execution_path = run_dir / "run.json"
        json_file(execution_path, {"ok": True, "status": "succeeded", "evidence_id": evidence_id,
                                   "lease_receipt": str(lease_path)})
        inputs = [
            {"path": str(self.contract.resolve()), "sha256": self.contract_sha},
            {"path": str(self.input_path.resolve()), "sha256": self.input_sha},
            {"path": str(self.oracle_path.resolve()), "sha256": self.oracle_sha},
            {"path": str(implementation["source_path"].resolve()), "sha256": implementation["source"]["sha256"]},
        ]
        if role == "profile_candidate":
            inputs.extend((
                {"path": str(self.repo_shim.resolve()), "sha256": self.repo_shim_sha},
                {"path": str(self.fake_ncu.resolve()), "sha256": self.fake_ncu_sha},
            ))
        implementation_name = (
            "baseline" if role in {"correctness_baseline", "baseline_timing"}
            else "candidate"
        )
        mode = "time" if role.endswith("timing") else "single" if role == "profile_candidate" else "correctness"
        argv = [
            str(implementation["binary_path"].relative_to(self.root)), "--case", "m16n16k16",
            "--input", str(self.input_path.relative_to(self.root)),
            "--impl", "cublas" if implementation_name == "baseline" else "ptx",
            "--mode", mode,
        ]
        if implementation_name == "candidate":
            argv.extend(("--stages", "2"))
        summary = {
            "status": "succeeded", "valid": True,
            "toolchain": {"root": "/opt/cuda/12.9", "version": "12.9",
                          "nvcc": "/opt/cuda/12.9/bin/nvcc",
                          "sanitizer": "/opt/cuda/12.9/bin/compute-sanitizer"},
            "provenance": {
                "source": {"project_root": str(self.root)},
                "inputs": inputs,
                "binary": {"path": str(implementation["binary_path"].resolve()),
                           "sha256": implementation["binary"]["sha256"]},
                "gpus": [{"uuid": self.gpu, "compute_capability": "7.0"}],
                "tools": {"nvcc": "Cuda compilation tools, release 12.9",
                          "ncu": {"path": str(self.fake_ncu), "version": "Nsight Compute CLI 2025.2.1"}},
                "argv": argv,
            },
            "binaries": [{"path": str(implementation["binary_path"].resolve()),
                          "sha256": implementation["binary"]["sha256"]}],
            "resource_samples": {"quiescence": {"uncontaminated": True}},
        }
        if timing:
            summary["benchmark"] = {"statistics": {"values": [2.1, 2.0, 2.2, 2.1, 2.0]}}
        exported = run_dir / "evidence.json"
        json_file(exported, {"evidence": {"id": evidence_id, "kind": kind,
                                           "status": "succeeded", "summary": summary}})
        return {
            "execution": {"path": str(execution_path.relative_to(self.base)), "sha256": CHECK.digest(execution_path)},
            "evidence": {"path": str(exported.relative_to(self.base)), "sha256": CHECK.digest(exported)},
            "lease": {"path": str(lease_path.relative_to(self.root)), "sha256": lease_sha},
        }

    def make_portable(self):
        """Move fixture bindings to a synthetic old root without copying runtime/tool binaries."""
        self.historical_root = Path(self.temp.name) / "captured" / "KernelWiki"
        self.assertFalse(self.historical_root.exists())
        self.manifest["source_root"] = str(self.historical_root)

        for name, entry in self.manifest["cases"][0]["implementations"].items():
            original = self.root / entry["binary"]["path"]
            archived = self.base / "bin" / f"{name}.archived"
            archived_sha = put(archived, original.read_bytes())
            entry["executed_path"] = str(original.relative_to(self.root))
            entry["binary"] = {"path": str(archived.relative_to(self.root)), "sha256": archived_sha}
            original.unlink()

        for role, run in self.runs.items():
            original_lease_path = self.root / run["lease"]["path"]
            lease_data = json.loads(original_lease_path.read_text())
            lease_data["project_root"] = str(self.historical_root)
            original_lease_sha = json_file(original_lease_path, lease_data)
            archived_lease = self.base / "controller" / role / "lease.json"
            archived_lease_sha = put(archived_lease, original_lease_path.read_bytes())
            self.assertEqual(original_lease_sha, archived_lease_sha)

            execution_path = self.base / run["execution"]["path"]
            execution = json.loads(execution_path.read_text())
            original_lease_identity = self.historical_root / original_lease_path.relative_to(self.root)
            execution["lease_receipt"] = str(original_lease_identity)
            run["execution"]["sha256"] = json_file(execution_path, execution)
            run["lease"] = {
                "path": str(archived_lease.relative_to(self.base)),
                "sha256": archived_lease_sha,
                "original_path": str(original_lease_identity),
            }

            evidence_path = self.base / run["evidence"]["path"]
            exported = json.loads(evidence_path.read_text())
            summary = exported["evidence"]["summary"]
            provenance = summary["provenance"]
            provenance["source"]["project_root"] = str(self.historical_root)
            for item in provenance["inputs"]:
                input_path = Path(item["path"])
                if role == "profile_candidate" and input_path.resolve() == self.fake_ncu.resolve():
                    item["path"] = "/captured-tools/ncu-not-installed"
                    continue
                try:
                    item["path"] = str(self.historical_root / input_path.relative_to(self.root))
                except ValueError:
                    pass
            binary_relative = Path(provenance["binary"]["path"]).relative_to(self.root)
            provenance["binary"]["path"] = str(self.historical_root / binary_relative)
            for binary in summary["binaries"]:
                binary_relative = Path(binary["path"]).relative_to(self.root)
                binary["path"] = str(self.historical_root / binary_relative)
            provenance["tools"]["ncu"]["path"] = "/captured-tools/ncu-not-installed"
            run["evidence"]["sha256"] = json_file(evidence_path, exported)

            if role == "profile_candidate":
                shim_log_path = self.base / run["profile"]["shim_log"]["path"]
                shim_record = json.loads(shim_log_path.read_text())
                shim_record["shim_path"] = str(self.historical_root / "benchmark_runs/volta-environment/ncu-collection-shim.py")
                shim_record["actual_ncu_path"] = "/captured-tools/ncu-not-installed"
                run["profile"]["shim_log"]["sha256"] = json_file(shim_log_path, shim_record)

        shutil.rmtree(self.root / ".todo-orchestrator")
        self.fake_ncu.unlink()
        self.write_manifest()

    def write_manifest(self):
        json_file(self.manifest_path, self.manifest)

    def validate(self):
        return CHECK.validate(self.manifest_path)

    def prepare_gemm_partial_case(self):
        contract = json.loads(self.contract.read_text())
        contract["numerical_contract"]["operator"] = CHECK.GEMM_OPERATOR
        contract["shape"] = {"m": 16, "n": 16, "k": 16}
        contract_sha = json_file(self.contract, contract)
        self.manifest["cases"][0]["contract"]["sha256"] = contract_sha
        contract_path = str(self.contract.resolve())
        for run in self.manifest["cases"][0]["runs"].values():
            exported_path = self.base / run["evidence"]["path"]
            exported = json.loads(exported_path.read_text())
            inputs = exported["evidence"]["summary"]["provenance"]["inputs"]
            for item in inputs:
                if item.get("path") == contract_path:
                    item["sha256"] = contract_sha
            run["evidence"]["sha256"] = json_file(exported_path, exported)
        native = {
            "format": CHECK.GEMM_CHECK_FORMAT,
            "case_id": "m16n16k16",
            "operator": "gemm",
            "implementation": "ptx",
            "shared_stages": 2,
            "passed": True,
            "checked_outputs": 16 * 16,
            "failed_outputs": 0,
            "nonfinite_outputs": 0,
            "max_abs_error": 0,
            "max_admission_tolerance": 0.004,
            "max_tolerance_ratio": 0,
        }
        sanitizer = {
            "tool": "compute-sanitizer", "status": "partial", "conclusive": False,
            "tool_detail": "memcheck", "exit_status": 0, "error_count": 0,
            "stdout_tail": [json.dumps(native)],
        }
        run = self.manifest["cases"][0]["runs"]["sanitizer_candidate"]
        run["sanitizer_summary"]["sha256"] = json_file(self.sanitizer_summary, sanitizer)
        self.write_manifest()

    def set_controller_argv(self, role, argv):
        run = self.manifest["cases"][0]["runs"][role]
        evidence_path = self.base / run["evidence"]["path"]
        exported = json.loads(evidence_path.read_text())
        exported["evidence"]["summary"]["provenance"]["argv"] = argv
        run["evidence"]["sha256"] = json_file(evidence_path, exported)
        self.write_manifest()

    def controller_argv(self, role):
        run = self.manifest["cases"][0]["runs"][role]
        evidence_path = self.base / run["evidence"]["path"]
        exported = json.loads(evidence_path.read_text())
        return exported["evidence"]["summary"]["provenance"]["argv"]

    def replace_argv_option(self, argv, flag, value):
        changed = list(argv)
        occurrences = [i for i, item in enumerate(changed) if item == flag]
        self.assertEqual(1, len(occurrences))
        changed[occurrences[0] + 1] = value
        return changed

    def rows_contract(self):
        cases = [
            {"case_id": "softmax_small", "operation": "softmax", "rows": 2, "cols": 2,
             "expected_outputs": 4, "baseline_path": "softmax_warp", "candidate_path": "softmax_cta",
             "check_identity": "logical-row-major:softmax_small"},
            {"case_id": "softmax_wide_streaming", "operation": "softmax", "rows": 1, "cols": 4,
             "expected_outputs": 4, "baseline_path": "softmax_warp", "candidate_path": "softmax_cta",
             "check_identity": "logical-row-major:softmax_wide_streaming"},
            {"case_id": "softmax_zero_rows", "operation": "softmax", "rows": 0, "cols": 3,
             "expected_outputs": 0, "baseline_path": "no_launch", "candidate_path": "no_launch",
             "check_identity": "logical-row-major:softmax_zero_rows"},
            {"case_id": "rmsnorm_vec4_wide_streaming", "operation": "rmsnorm", "rows": 1, "cols": 4,
             "expected_outputs": 4, "baseline_path": "rmsnorm_cta_scalar", "candidate_path": "vec4",
             "check_identity": "logical-row-major:rmsnorm_vec4_wide_streaming"},
            {"case_id": "rmsnorm_misaligned", "operation": "rmsnorm", "rows": 1, "cols": 3,
             "expected_outputs": 3, "baseline_path": "rmsnorm_cta_scalar", "candidate_path": "scalar_fallback",
             "check_identity": "logical-row-major:rmsnorm_misaligned"},
        ]
        ids = [item["case_id"] for item in cases]
        return {
            "format": CHECK.ROW_CONTRACT_FORMAT, "case_id": "rows",
            "suite_id": "rows_correctness_suite", "operators": ["softmax", "rmsnorm"],
            "case_ids": ids, "correctness_case_ids": ids, "cases": cases,
            "selected_timing_case_ids": ["softmax_wide_streaming", "rmsnorm_vec4_wide_streaming"],
            "numerical_contract": {"common": {"tolerance_atol": 2.0 ** -30,
                                                "tolerance_rtol": 2.0 ** -14}},
        }

    def rows_native_result(self, contract, implementation="candidate"):
        path_key = "baseline_path" if implementation == "baseline" else "candidate_path"
        checks = []
        for spec in contract["cases"]:
            checks.append({
                "case_id": spec["case_id"], "operation": spec["operation"],
                "rows": spec["rows"], "cols": spec["cols"],
                "expected_outputs": spec["expected_outputs"],
                "implementation_path": spec[path_key], "check_identity": spec["check_identity"],
                "checked_outputs": spec["expected_outputs"], "failed_outputs": 0,
                "nonfinite_outputs": 0, "padding_checked": 4, "padding_writes": 0,
                "failed_checks": 0, "max_abs_error": 0.0,
                "max_admission_tolerance": 0.01, "max_tolerance_ratio": 0.0,
                "atol": 2.0 ** -30, "rtol": 2.0 ** -14, "passed": True,
            })
        total = sum(item["expected_outputs"] for item in contract["cases"])
        return {
            "format": CHECK.ROW_CHECK_FORMAT, "status": "passed", "gpu_uuid": "00000000000000000000000000000001",
            "compute_capability": "7.0", "operator": "row_suite", "implementation": implementation,
            "case_scope": "full_suite", "case_count": len(checks),
            "expected_outputs": total, "checked_outputs": total,
            "failed_outputs": 0, "nonfinite_outputs": 0, "padding_writes": 0,
            "cases": checks,
        }

    def rows_role_summaries(self, contract, input_path, reference_path, reference_sha):
        timing_cases = ",".join(contract["selected_timing_case_ids"])
        summaries = {}
        for role, (implementation, mode, selection, warmups, repetitions, rotations) in CHECK.ROW_PRIMARY_ROLES.items():
            argv = [str(self.impl[implementation]["binary_path"].relative_to(self.root)),
                    "--input", str(input_path.relative_to(self.root)),
                    "--reference", str(reference_path.relative_to(self.root)),
                    "--impl", implementation, "--mode", mode]
            if selection == "timing":
                argv.extend(("--case", timing_cases))
            elif selection == "profile":
                argv.extend(("--case", CHECK.ROW_PROFILE_CASE_ID))
            if warmups is not None:
                argv.extend(("--warmups", str(warmups), "--repetitions", str(repetitions),
                             "--rotations", str(rotations)))
            summaries[role] = {"provenance": {
                "source": {"project_root": str(self.root)},
                "argv": argv,
                "binary": {"path": str(self.impl[implementation]["binary_path"].resolve())},
                "inputs": [{"path": str(reference_path.resolve()), "sha256": reference_sha}],
            }}
        return summaries

    def rows_timing_result(self, contract, role):
        implementation, _mode, selection, warmups, repetitions, rotations = CHECK.ROW_PRIMARY_ROLES[role]
        ids = contract["selected_timing_case_ids"] if selection == "timing" else [CHECK.ROW_PROFILE_CASE_ID]
        specs = {item["case_id"]: item for item in contract["cases"]}
        path_key = "baseline_path" if implementation == "baseline" else "candidate_path"
        return {
            "format": CHECK.ROW_TIMING_FORMAT, "status": "completed",
            "gpu_uuid": "00000000000000000000000000000001", "compute_capability": "7.0",
            "implementation": implementation, "warmups": warmups, "repetitions": repetitions,
            "rotation_count": rotations,
            "cases": [{"case_id": case_id, "operation": specs[case_id]["operation"],
                       "rows": specs[case_id]["rows"], "cols": specs[case_id]["cols"],
                       "implementation_path": specs[case_id][path_key],
                       "samples_us": [2.5] * repetitions}
                      for case_id in ids],
        }

    def test_complete_matched_evidence_passes(self):
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["checked_cases"], 1)

    def test_not_run_cannot_claim_evaluated_not_promoted(self):
        self.manifest["cases"][0]["disposition"] = "not_run"
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("not_run is incomplete", result["errors"][0])

    def test_missing_controller_receipt_fails(self):
        del self.manifest["cases"][0]["runs"]["candidate_timing"]
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("controller receipt is missing", result["errors"][0])

    def test_case_cli_uses_the_canonical_family_manifest_map(self):
        self.assertEqual(CHECK.CASE_PATHS["environment"], "benchmark_runs/volta-environment/summary.json")
        self.assertEqual(CHECK.CASE_PATHS["gemm"], "benchmark_runs/volta-gemm/summary.json")
        self.assertEqual(CHECK.CASE_PATHS["grouped-lowbit"], "benchmark_runs/volta-grouped-lowbit/summary.json")
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--case", result.stdout)
        self.assertIn("--all", result.stdout)

    def test_execution_must_bind_the_hashed_lease_receipt(self):
        execution = self.base / self.runs["correctness_candidate"]["execution"]["path"]
        execution_data = json.loads(execution.read_text())
        execution_data.pop("lease_receipt")
        self.manifest["cases"][0]["runs"]["correctness_candidate"]["execution"]["sha256"] = json_file(
            execution, execution_data)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("lease path differs from execution receipt", result["errors"][0])

    def test_failed_run_fails_even_when_disposition_claims_evaluation(self):
        execution = self.base / self.runs["correctness_candidate"]["execution"]["path"]
        execution_data = json.loads(execution.read_text())
        execution_data["ok"] = False
        self.manifest["cases"][0]["runs"]["correctness_candidate"]["execution"]["sha256"] = json_file(execution, execution_data)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("did not succeed", result["errors"][0])

    def test_controller_source_input_must_be_bound(self):
        evidence = self.base / self.runs["candidate_timing"]["evidence"]["path"]
        exported = json.loads(evidence.read_text())
        exported["evidence"]["summary"]["provenance"]["inputs"] = []
        self.manifest["cases"][0]["runs"]["candidate_timing"]["evidence"]["sha256"] = json_file(evidence, exported)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("do not bind", result["errors"][0])

    def install_archived_source_binding(self):
        original_path = "artifacts/kernels/example/frozen/source.cu"
        archive_path = self.root / "benchmark_runs" / "volta-case" / "source-snapshot" / "source.cu"
        source_sha = put(archive_path, b"captured source bytes\n")
        self.assertFalse((self.root / original_path).exists())
        source_record = {
            "path": str(archive_path.relative_to(self.root)),
            "original_path": original_path,
            "sha256": source_sha,
        }
        for name in ("baseline", "candidate"):
            self.impl[name]["source"] = dict(source_record)
            self.impl[name]["source_path"] = archive_path
            self.manifest["cases"][0]["implementations"][name]["source"] = dict(source_record)
        for role, run in self.runs.items():
            evidence_path = self.base / run["evidence"]["path"]
            exported = json.loads(evidence_path.read_text())
            inputs = exported["evidence"]["summary"]["provenance"]["inputs"]
            inputs[3] = {"path": str((self.root / original_path).resolve()), "sha256": source_sha}
            run["evidence"]["sha256"] = json_file(evidence_path, exported)
        self.manifest["cases"][0]["captured_input_archives"] = [{
            "original_path": original_path,
            "archive_path": str(archive_path.relative_to(self.root)),
            "sha256": source_sha,
        }]
        self.write_manifest()

    def test_archived_source_is_bound_to_missing_captured_path_and_legacy_stays_valid(self):
        legacy = self.validate()
        self.assertTrue(legacy["valid"], legacy["errors"])
        self.assertNotIn("original_path", self.manifest["cases"][0]["implementations"]["candidate"]["source"])

        self.install_archived_source_binding()
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])

    def test_archived_source_rejects_wrong_or_escaping_original_path(self):
        self.install_archived_source_binding()
        source = self.manifest["cases"][0]["implementations"]["baseline"]["source"]
        source["original_path"] = "artifacts/kernels/example/frozen/other.cu"
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("does not match its captured input archive", result["errors"][0])

        self.install_archived_source_binding()
        source = self.manifest["cases"][0]["implementations"]["baseline"]["source"]
        source["original_path"] = "../outside/source.cu"
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("safe repository-relative path", result["errors"][0])

    def test_archived_source_rejects_archive_hash_and_controller_alias_mismatches(self):
        self.install_archived_source_binding()
        self.manifest["cases"][0]["captured_input_archives"][0]["sha256"] = "0" * 64
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("archive SHA-256 mismatch", result["errors"][0])

        self.install_archived_source_binding()
        source_archive = self.manifest["cases"][0]["captured_input_archives"][0]
        archive_path = self.root / "data" / "outside-evidence-archive.txt"
        put(archive_path, (self.root / source_archive["archive_path"]).read_bytes())
        source_archive["archive_path"] = str(archive_path.relative_to(self.root))
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("archive path escapes the family evidence directory", result["errors"][0])

        self.install_archived_source_binding()
        run = self.manifest["cases"][0]["runs"]["correctness_candidate"]
        evidence_path = self.base / run["evidence"]["path"]
        exported = json.loads(evidence_path.read_text())
        exported["evidence"]["summary"]["provenance"]["inputs"][-1]["sha256"] = "0" * 64
        run["evidence"]["sha256"] = json_file(evidence_path, exported)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("do not bind the contract, represented data, and source", result["errors"][0])

        self.install_archived_source_binding()
        run = self.manifest["cases"][0]["runs"]["correctness_candidate"]
        evidence_path = self.base / run["evidence"]["path"]
        exported = json.loads(evidence_path.read_text())
        inputs = exported["evidence"]["summary"]["provenance"]["inputs"]
        inputs.append(dict(inputs[-1]))
        run["evidence"]["sha256"] = json_file(evidence_path, exported)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("exactly bind archived source aliases", result["errors"][0])

    def test_portable_source_alias_archive_remains_bound_after_relocation(self):
        self.install_archived_source_binding()
        self.make_portable()
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])

    def test_controller_binary_must_match_declared_artifact(self):
        evidence = self.base / self.runs["candidate_timing"]["evidence"]["path"]
        exported = json.loads(evidence.read_text())
        exported["evidence"]["summary"]["binaries"][0]["sha256"] = "0" * 64
        self.manifest["cases"][0]["runs"]["candidate_timing"]["evidence"]["sha256"] = json_file(evidence, exported)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("not bound to binary", result["errors"][0])

    def test_archived_binary_must_bind_original_executed_path_and_hash(self):
        candidate = self.manifest["cases"][0]["implementations"]["candidate"]
        original = candidate["binary"]["path"]
        original_path = self.root / original
        archived_path = self.base / "bin" / "candidate-hash.bin"
        archived_path.parent.mkdir(parents=True, exist_ok=True)
        archived_path.write_bytes(original_path.read_bytes())
        candidate["executed_path"] = original
        candidate["binary"] = {
            "path": str(archived_path.relative_to(self.root)),
            "sha256": hashlib.sha256(archived_path.read_bytes()).hexdigest(),
        }
        self.write_manifest()
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])

        candidate["executed_path"] = self.impl["baseline"]["binary"]["path"]
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("not bound to binary", result["errors"][0])

    def test_portable_archive_passes_without_runtime_tree_executed_elf_or_ncu(self):
        self.reset_fixture()
        self.make_portable()
        self.assertFalse((self.root / ".todo-orchestrator").exists())
        self.assertFalse(self.historical_root.exists())
        self.assertFalse(Path("/captured-tools/ncu-not-installed").exists())
        for entry in self.manifest["cases"][0]["implementations"].values():
            self.assertTrue((self.root / entry["binary"]["path"]).is_file())
            self.assertFalse((self.historical_root / entry["executed_path"]).exists())
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])

    def test_portable_lease_copy_and_execution_original_path_are_bound(self):
        self.reset_fixture()
        self.make_portable()
        run = self.manifest["cases"][0]["runs"]["sanitizer_candidate"]
        lease_path = self.base / run["lease"]["path"]
        lease_path.write_bytes(lease_path.read_bytes() + b"tampered\n")
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("archived lease: SHA-256 mismatch", result["errors"][0])

        self.reset_fixture()
        self.make_portable()
        run = self.manifest["cases"][0]["runs"]["sanitizer_candidate"]
        execution_path = self.base / run["execution"]["path"]
        execution = json.loads(execution_path.read_text())
        execution["lease_receipt"] += ".wrong"
        run["execution"]["sha256"] = json_file(execution_path, execution)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("lease path differs from execution receipt", result["errors"][0])

    def test_portable_source_root_and_input_remapping_are_bound(self):
        self.reset_fixture()
        self.make_portable()
        run = self.manifest["cases"][0]["runs"]["correctness_baseline"]
        evidence_path = self.base / run["evidence"]["path"]
        exported = json.loads(evidence_path.read_text())
        exported["evidence"]["summary"]["provenance"]["source"]["project_root"] = str(
            self.historical_root / "other")
        run["evidence"]["sha256"] = json_file(evidence_path, exported)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("source snapshot is unbound", result["errors"][0])

        self.reset_fixture()
        self.make_portable()
        run = self.manifest["cases"][0]["runs"]["correctness_baseline"]
        evidence_path = self.base / run["evidence"]["path"]
        exported = json.loads(evidence_path.read_text())
        inputs = exported["evidence"]["summary"]["provenance"]["inputs"]
        inputs[0]["path"] = str(self.historical_root / "../escaped/contract.json")
        run["evidence"]["sha256"] = json_file(evidence_path, exported)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("escapes historical source_root", result["errors"][0])

    def test_portable_manifest_rejects_explicit_null_source_root(self):
        self.manifest["source_root"] = None
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("source_root must be an absolute historical path", result["errors"][0])

    def test_portable_archived_elf_hash_and_historical_identity_are_bound(self):
        self.make_portable()
        candidate = self.manifest["cases"][0]["implementations"]["candidate"]
        archived = self.root / candidate["binary"]["path"]
        archived.write_bytes(archived.read_bytes() + b"tampered")
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("SHA-256 mismatch", result["errors"][0])

        self.reset_fixture()
        self.make_portable()
        run = self.manifest["cases"][0]["runs"]["candidate_timing"]
        evidence_path = self.base / run["evidence"]["path"]
        exported = json.loads(evidence_path.read_text())
        exported["evidence"]["summary"]["binaries"][0]["path"] += ".wrong"
        run["evidence"]["sha256"] = json_file(evidence_path, exported)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("not bound to binary", result["errors"][0])

    def test_portable_executable_input_uses_archived_role_elf_not_mutable_checkout_path(self):
        self.make_portable()
        candidate = self.manifest["cases"][0]["implementations"]["candidate"]
        mutable_checkout_path = self.root / candidate["executed_path"]
        put(mutable_checkout_path, b"different file now at captured executable path")
        run = self.manifest["cases"][0]["runs"]["correctness_candidate"]
        evidence_path = self.base / run["evidence"]["path"]
        exported = json.loads(evidence_path.read_text())
        provenance = exported["evidence"]["summary"]["provenance"]
        binary_identity = str(self.historical_root / candidate["executed_path"])
        provenance["inputs"].append({
            "path": binary_identity, "sha256": candidate["binary"]["sha256"],
        })
        run["evidence"]["sha256"] = json_file(evidence_path, exported)
        self.write_manifest()
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])

        evidence_path = self.base / run["evidence"]["path"]
        exported = json.loads(evidence_path.read_text())
        provenance = exported["evidence"]["summary"]["provenance"]
        binary_input = next(item for item in provenance["inputs"] if item["path"] == binary_identity)
        binary_input["sha256"] = "0" * 64
        run["evidence"]["sha256"] = json_file(evidence_path, exported)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("executed binary input differs from its archived ELF", result["errors"][0])

    def test_portable_unarchived_repo_input_must_match_current_checkout(self):
        self.make_portable()
        input_file = self.root / "docs" / "captured-extra.txt"
        input_sha = put(input_file, b"source bytes at capture time")
        input_file.write_bytes(b"different current checkout bytes")
        run = self.manifest["cases"][0]["runs"]["correctness_baseline"]
        evidence_path = self.base / run["evidence"]["path"]
        exported = json.loads(evidence_path.read_text())
        exported["evidence"]["summary"]["provenance"]["inputs"].append({
            "path": str(self.historical_root / "docs" / input_file.name), "sha256": input_sha,
        })
        run["evidence"]["sha256"] = json_file(evidence_path, exported)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("repository input is absent or mismatched and has no archive", result["errors"][0])

    def test_portable_archived_repo_input_closes_missing_source_path(self):
        self.make_portable()
        original = self.root / "docs" / "frozen-extra.txt"
        input_sha = put(original, b"frozen source record")
        archive = self.base / "source-archive" / "frozen-extra.txt"
        put(archive, original.read_bytes())
        original.unlink()
        self.manifest["cases"][0]["captured_input_archives"] = [{
            "original_path": "docs/frozen-extra.txt",
            "archive_path": str(archive.relative_to(self.root)),
            "sha256": input_sha,
        }]
        for run in self.manifest["cases"][0]["runs"].values():
            evidence_path = self.base / run["evidence"]["path"]
            exported = json.loads(evidence_path.read_text())
            exported["evidence"]["summary"]["provenance"]["inputs"].append({
                "path": str(self.historical_root / "docs" / original.name), "sha256": input_sha,
            })
            run["evidence"]["sha256"] = json_file(evidence_path, exported)
        self.write_manifest()
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])

    def test_portable_ncu_hash_and_path_remain_capture_bound_without_binary(self):
        self.make_portable()
        self.assertFalse(Path("/captured-tools/ncu-not-installed").exists())
        run = self.manifest["cases"][0]["runs"]["profile_candidate"]
        shim_log_path = self.base / run["profile"]["shim_log"]["path"]
        shim_record = json.loads(shim_log_path.read_text())
        shim_record["actual_ncu_sha256"] = "0" * 64
        run["profile"]["shim_log"]["sha256"] = json_file(shim_log_path, shim_record)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("no-clock/source-import policy", result["errors"][0])

    def test_portable_ncu_identity_requires_absolute_captured_path(self):
        self.make_portable()
        run = self.manifest["cases"][0]["runs"]["profile_candidate"]
        evidence_path = self.base / run["evidence"]["path"]
        exported = json.loads(evidence_path.read_text())
        exported["evidence"]["summary"]["provenance"]["tools"]["ncu"]["path"] = "tools/ncu"
        run["evidence"]["sha256"] = json_file(evidence_path, exported)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("NCU tool path must be absolute", result["errors"][0])

        self.reset_fixture()
        self.make_portable()
        run = self.manifest["cases"][0]["runs"]["profile_candidate"]
        evidence_path = self.base / run["evidence"]["path"]
        exported = json.loads(evidence_path.read_text())
        exported["evidence"]["summary"]["provenance"]["tools"]["ncu"]["path"] = "/captured-tools/other-ncu"
        run["evidence"]["sha256"] = json_file(evidence_path, exported)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("no-clock/source-import policy", result["errors"][0])

    def test_declared_library_must_be_bound_in_controller_inputs(self):
        library = self.root / "lib" / "libbaseline.so"
        library_sha = put(library, b"synthetic baseline library")
        self.manifest["cases"][0]["libraries"] = [{
            "implementation": "baseline", "name": "baseline-math", "version": "1.0",
            "path": str(library), "sha256": library_sha,
        }]
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("do not bind declared library", result["errors"][0])

        for role in ("correctness_baseline", "baseline_timing"):
            evidence = self.base / self.runs[role]["evidence"]["path"]
            exported = json.loads(evidence.read_text())
            exported["evidence"]["summary"]["provenance"]["inputs"].append(
                {"path": str(library.resolve()), "sha256": library_sha})
            self.manifest["cases"][0]["runs"][role]["evidence"]["sha256"] = json_file(evidence, exported)
        self.write_manifest()
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])

    def test_portable_external_library_identity_does_not_require_vendor_file(self):
        self.make_portable()
        library_path = "/captured-libraries/libcublas.so"
        self.assertFalse(Path(library_path).exists())
        library_sha = "a" * 64
        self.manifest["cases"][0]["libraries"] = [{
            "implementation": "baseline", "name": "cuBLAS", "version": "12.9",
            "path": library_path, "sha256": "malformed",
        }]
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("external library identity has a malformed SHA-256", result["errors"][0])

        self.manifest["cases"][0]["libraries"][0]["sha256"] = library_sha
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("do not bind declared library", result["errors"][0])

        for role in ("correctness_baseline", "baseline_timing"):
            evidence = self.base / self.runs[role]["evidence"]["path"]
            exported = json.loads(evidence.read_text())
            inputs = exported["evidence"]["summary"]["provenance"]["inputs"]
            inputs[:] = [item for item in inputs if item.get("path") != library_path]
            inputs.append({"path": library_path, "sha256": "b" * 64})
            self.manifest["cases"][0]["runs"][role]["evidence"]["sha256"] = json_file(evidence, exported)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("do not bind declared library", result["errors"][0])

        for role in ("correctness_baseline", "baseline_timing"):
            evidence = self.base / self.runs[role]["evidence"]["path"]
            exported = json.loads(evidence.read_text())
            inputs = exported["evidence"]["summary"]["provenance"]["inputs"]
            inputs[:] = [item for item in inputs if item.get("path") != library_path]
            inputs.append({"path": library_path, "sha256": library_sha})
            self.manifest["cases"][0]["runs"][role]["evidence"]["sha256"] = json_file(evidence, exported)
        self.write_manifest()
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])

    def test_portable_repo_library_archive_is_hashed_after_remapping(self):
        self.make_portable()
        library = self.root / "lib" / "libcutlass-header-identity.txt"
        library_sha = put(library, b"repo-contained CUTLASS header archive")
        historical_library_path = str(self.historical_root / "lib" / library.name)
        self.manifest["cases"][0]["libraries"] = [{
            "implementation": "baseline", "name": "CUTLASS headers", "version": "2.11",
            "path": historical_library_path, "sha256": library_sha,
        }]
        for role in ("correctness_baseline", "baseline_timing"):
            evidence = self.base / self.runs[role]["evidence"]["path"]
            exported = json.loads(evidence.read_text())
            exported["evidence"]["summary"]["provenance"]["inputs"].append(
                {"path": historical_library_path, "sha256": library_sha})
            self.manifest["cases"][0]["runs"][role]["evidence"]["sha256"] = json_file(evidence, exported)
        self.write_manifest()
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])

        library.write_bytes(library.read_bytes() + b"tampered")
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("linked library archive is unavailable or mismatched", result["errors"][0])

    def test_portable_repo_library_can_use_captured_archive_when_checkout_path_is_absent(self):
        self.make_portable()
        original = self.root / "lib" / "captured-cutlass-header.txt"
        library_sha = put(original, b"frozen CUTLASS include identity")
        archive = self.base / "source-archive" / "captured-cutlass-header.txt"
        put(archive, original.read_bytes())
        original.unlink()
        historical_path = str(self.historical_root / "lib" / original.name)
        self.manifest["cases"][0]["libraries"] = [{
            "implementation": "baseline", "name": "CUTLASS headers", "version": "2.11",
            "path": historical_path, "sha256": library_sha,
        }]
        self.manifest["cases"][0]["captured_input_archives"] = [{
            "original_path": "lib/captured-cutlass-header.txt",
            "archive_path": str(archive.relative_to(self.root)),
            "sha256": library_sha,
        }]
        for run in self.manifest["cases"][0]["runs"].values():
            evidence_path = self.base / run["evidence"]["path"]
            exported = json.loads(evidence_path.read_text())
            exported["evidence"]["summary"]["provenance"]["inputs"].append({
                "path": historical_path, "sha256": library_sha,
            })
            run["evidence"]["sha256"] = json_file(evidence_path, exported)
        self.write_manifest()
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])

    def test_oracle_rejects_negative_and_boolean_numeric_evidence(self):
        for key, value in (("max_abs_error", -100.0), ("atol", -1.0), ("reference_max_abs", True)):
            with self.subTest(key=key, value=value):
                oracle = json.loads(self.oracle_path.read_text())
                oracle["checks"][0][key] = value
                self.manifest["cases"][0]["oracle"]["sha256"] = json_file(self.oracle_path, oracle)
                self.write_manifest()
                result = self.validate()
                self.assertFalse(result["valid"])
                self.assertTrue(any("CPU oracle" in error for error in result["errors"]))

    def test_sanitizer_errors_fail(self):
        self.manifest["cases"][0]["runs"]["sanitizer_candidate"]["sanitizer_summary"]["sha256"] = json_file(
            self.sanitizer_summary, {"status": "passed", "error_count": 1})
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("sanitizer reported errors", result["errors"][0])

    def test_partial_memcheck_requires_zero_errors_and_bound_native_pass(self):
        raw_uuid = self.gpu.removeprefix("GPU-").replace("-", "")
        partial = {
            "tool": "compute-sanitizer", "status": "partial", "conclusive": False,
            "tool_detail": "memcheck", "exit_status": 0, "error_count": 0,
            "stdout_tail": [json.dumps({"status": "passed", "compute_capability": "7.0",
                                         "gpu_uuid": raw_uuid})],
        }
        self.manifest["cases"][0]["runs"]["sanitizer_candidate"]["sanitizer_summary"]["sha256"] = json_file(
            self.sanitizer_summary, partial)
        self.write_manifest()
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])

        partial["stdout_tail"] = ["not a native PASS record"]
        self.manifest["cases"][0]["runs"]["sanitizer_candidate"]["sanitizer_summary"]["sha256"] = json_file(
            self.sanitizer_summary, partial)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("sanitizer reported errors or incomplete execution", result["errors"][0])

    def test_partial_memcheck_rejects_boolean_zero_fields(self):
        raw_uuid = self.gpu.removeprefix("GPU-").replace("-", "")
        partial = {
            "tool": "compute-sanitizer", "status": "partial", "conclusive": False,
            "tool_detail": "memcheck", "exit_status": 0, "error_count": 0,
            "stdout_tail": [json.dumps({"status": "passed", "compute_capability": "7.0",
                                         "gpu_uuid": raw_uuid})],
        }
        for field in ("exit_status", "error_count"):
            with self.subTest(field=field):
                partial[field] = False
                self.manifest["cases"][0]["runs"]["sanitizer_candidate"]["sanitizer_summary"]["sha256"] = json_file(
                    self.sanitizer_summary, partial)
                self.write_manifest()
                result = self.validate()
                self.assertFalse(result["valid"])
                self.assertIn("sanitizer reported errors or incomplete execution", result["errors"][0])
                partial[field] = 0

    def test_partial_memcheck_accepts_contract_bound_gemm_native_pass(self):
        contract = {
            "case_id": "large_dense",
            "numerical_contract": {"operator": CHECK.GEMM_OPERATOR},
            "shape": {"m": 1024, "n": 1024, "k": 1024},
        }
        native = {
            "format": CHECK.GEMM_CHECK_FORMAT,
            "case_id": "large_dense",
            "operator": "gemm",
            "implementation": "ptx",
            "shared_stages": 2,
            "passed": True,
            "checked_outputs": 1024 * 1024,
            "failed_outputs": 0,
            "nonfinite_outputs": 0,
            "max_abs_error": 0,
            "max_admission_tolerance": 0.0045236260260969833,
            "max_tolerance_ratio": 0,
        }
        partial = {
            "tool": "compute-sanitizer", "status": "partial", "conclusive": False,
            "tool_detail": "memcheck", "exit_status": 0, "error_count": 0,
            "stdout_tail": [json.dumps(native)],
        }
        self.assertTrue(CHECK._partial_memcheck_has_application_pass(partial, self.gpu, contract))

    def test_gemm_partial_native_pass_requires_controller_role_selectors(self):
        self.prepare_gemm_partial_case()
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])

    def test_gemm_controller_argv_rejects_wrong_implementation(self):
        self.prepare_gemm_partial_case()
        argv = self.replace_argv_option(
            self.controller_argv("sanitizer_candidate"), "--impl", "cublas"
        )
        self.set_controller_argv("sanitizer_candidate", argv)
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertTrue(any("--impl selector does not match" in error for error in result["errors"]))

    def test_gemm_controller_argv_rejects_wrong_stage_count(self):
        self.prepare_gemm_partial_case()
        argv = self.replace_argv_option(
            self.controller_argv("candidate_timing"), "--stages", "1"
        )
        self.set_controller_argv("candidate_timing", argv)
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertTrue(any("--stages selector does not match" in error for error in result["errors"]))

    def test_gemm_controller_argv_rejects_wrong_mode(self):
        self.prepare_gemm_partial_case()
        argv = self.replace_argv_option(
            self.controller_argv("profile_candidate"), "--mode", "time"
        )
        self.set_controller_argv("profile_candidate", argv)
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertTrue(any("--mode selector does not match" in error for error in result["errors"]))

    def test_gemm_controller_argv_rejects_wrong_case(self):
        self.prepare_gemm_partial_case()
        argv = self.replace_argv_option(
            self.controller_argv("profile_candidate"), "--case", "other-case"
        )
        self.set_controller_argv("profile_candidate", argv)
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertTrue(any("--case selector does not match" in error for error in result["errors"]))

    def test_gemm_controller_argv_rejects_wrong_input(self):
        self.prepare_gemm_partial_case()
        argv = self.replace_argv_option(
            self.controller_argv("profile_candidate"), "--input",
            "benchmark_runs/volta-case/inputs/wrong.bin"
        )
        self.set_controller_argv("profile_candidate", argv)
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertTrue(any("--input does not resolve to represented GEMM input" in error
                            for error in result["errors"]))

    def test_gemm_controller_argv_rejects_duplicate_selectors(self):
        self.prepare_gemm_partial_case()
        argv = self.controller_argv("sanitizer_candidate") + ["--impl", "cublas"]
        self.set_controller_argv("sanitizer_candidate", argv)
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertTrue(any("must contain 1 --impl selector" in error for error in result["errors"]), result["errors"])

    def test_gemm_controller_argv_rejects_duplicate_input_selectors(self):
        self.prepare_gemm_partial_case()
        argv = self.controller_argv("sanitizer_candidate") + [
            "--input", "benchmark_runs/volta-case/inputs/represented.bin"
        ]
        self.set_controller_argv("sanitizer_candidate", argv)
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertTrue(any("exactly one --input selector" in error for error in result["errors"]), result["errors"])

    def test_gemm_controller_argv_rejects_unbound_executable(self):
        self.prepare_gemm_partial_case()
        argv = list(self.controller_argv("sanitizer_candidate"))
        argv[0] = "/tmp/unrelated-gemm-runner"
        self.set_controller_argv("sanitizer_candidate", argv)
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertTrue(any("argv[0] differs from the declared GEMM executable" in error
                            for error in result["errors"]), result["errors"])

    def test_partial_memcheck_rejects_unbound_or_incomplete_gemm_native_pass(self):
        contract = {
            "case_id": "large_dense",
            "numerical_contract": {"operator": CHECK.GEMM_OPERATOR},
            "shape": {"m": 1024, "n": 1024, "k": 1024},
        }
        native = {
            "format": CHECK.GEMM_CHECK_FORMAT,
            "case_id": "large_dense",
            "operator": "gemm",
            "implementation": "ptx",
            "shared_stages": 2,
            "passed": True,
            "checked_outputs": 1024 * 1024,
            "failed_outputs": 0,
            "nonfinite_outputs": 0,
            "max_abs_error": 0,
            "max_admission_tolerance": 0.0045236260260969833,
            "max_tolerance_ratio": 0,
        }
        partial = {
            "tool": "compute-sanitizer", "status": "partial", "conclusive": False,
            "tool_detail": "memcheck", "exit_status": 0, "error_count": 0,
            "stdout_tail": [],
        }
        invalid_results = []
        for field, value in (
            ("format", "KERNELWIKI-OTHER-CHECK/1"),
            ("case_id", "small_m"),
            ("implementation", "cublas"),
            ("shared_stages", 1),
            ("checked_outputs", 1024 * 1024 - 1),
            ("failed_outputs", True),
            ("nonfinite_outputs", None),
            ("failed_outputs", 1),
            ("passed", False),
            ("max_tolerance_ratio", 1.000001),
            ("max_abs_error", 1000),
            ("max_abs_error", -1),
            ("max_admission_tolerance", True),
            ("max_admission_tolerance", float("inf")),
        ):
            changed = json.loads(json.dumps(native))
            if value is None:
                changed.pop(field)
            else:
                changed[field] = value
            invalid_results.append((field, changed))
        for field, changed in invalid_results:
            with self.subTest(field=field, value=changed.get(field)):
                partial["stdout_tail"] = [json.dumps(changed)]
                self.assertFalse(CHECK._partial_memcheck_has_application_pass(
                    partial, self.gpu, contract
                ))

        for invalid_shape in ({"m": True, "n": 1024}, {"m": 0, "n": 1024}):
            bad_contract = json.loads(json.dumps(contract))
            bad_contract["shape"] = invalid_shape
            partial["stdout_tail"] = [json.dumps(native)]
            with self.subTest(shape=invalid_shape):
                self.assertFalse(CHECK._partial_memcheck_has_application_pass(
                    partial, self.gpu, bad_contract
                ))

    def test_rows_native_receipt_binds_full_mixed_suite_and_declared_routes(self):
        contract = self.rows_contract()
        for implementation in ("baseline", "candidate"):
            native = self.rows_native_result(contract, implementation)
            self.assertTrue(CHECK._rows_native_pass_matches_contract(
                native, contract, self.gpu, implementation
            ))
        native = self.rows_native_result(contract)
        partial = {
            "tool": "compute-sanitizer", "status": "partial", "conclusive": False,
            "tool_detail": "memcheck", "exit_status": 0, "error_count": 0,
            "stdout_tail": [json.dumps(native)],
        }
        self.assertTrue(CHECK._partial_memcheck_has_application_pass(partial, self.gpu, contract))

        invalid = []
        for field, value in (
            ("status", "failed"), ("implementation", "baseline"),
            ("case_scope", "selected_subset"), ("gpu_uuid", "0" * 32),
            ("case_count", True), ("checked_outputs", 14),
            ("failed_outputs", True), ("nonfinite_outputs", 1),
        ):
            changed = json.loads(json.dumps(native))
            changed[field] = value
            invalid.append(changed)
        changed_case = json.loads(json.dumps(native))
        changed_case["cases"][0]["implementation_path"] = "other_route"
        invalid.append(changed_case)
        changed_case = json.loads(json.dumps(native))
        changed_case["cases"][0]["check_identity"] = "other-output"
        invalid.append(changed_case)
        changed_case = json.loads(json.dumps(native))
        changed_case["cases"][0]["max_tolerance_ratio"] = 1.01
        invalid.append(changed_case)
        changed_case = json.loads(json.dumps(native))
        changed_case["cases"][0]["rows"] = True
        invalid.append(changed_case)
        for changed in invalid:
            with self.subTest(native=changed):
                self.assertFalse(CHECK._rows_native_pass_matches_contract(
                    changed, contract, self.gpu, "candidate"
                ))

        for status, exit_status, error_count in (("partial", 1, 0), ("partial", 0, None),
                                                  ("passed", 0, 0)):
            bad_partial = dict(partial, status=status, exit_status=exit_status,
                               error_count=error_count, stdout_tail=[json.dumps(native)])
            self.assertFalse(CHECK._partial_memcheck_has_application_pass(
                bad_partial, self.gpu, contract
            ))

    def test_rows_contract_case_declarations_match_cpu_oracle(self):
        contract = self.rows_contract()
        oracle = {"checks": [{
            "case_id": item["case_id"], "operation": item["operation"],
            "rows": item["rows"], "cols": item["cols"],
            "reference_values": item["expected_outputs"], "passed": True,
        } for item in contract["cases"]]}
        self.assertEqual(contract["cases"], CHECK._validate_rows_oracle_contract(contract, oracle))
        oracle["checks"][1]["cols"] = True
        with self.assertRaisesRegex(CHECK.EvidenceError, "does not match contract case"):
            CHECK._validate_rows_oracle_contract(contract, oracle)
        malformed = json.loads(json.dumps(contract))
        malformed["cases"][0]["operation"] = []
        with self.assertRaises(CHECK.EvidenceError):
            CHECK._row_contract_cases(malformed)

    def test_rows_contract_binds_exact_input_reference_and_oracle_files(self):
        reference_path = self.base / "inputs" / "rows-reference.bin"
        reference_sha = put(reference_path, b"row reference bytes")
        oracle = {"format": "KERNELWIKI-CPU-ORACLE/1", "passed": True,
                  "represented_input_sha256": self.input_sha,
                  "reference_sha256": reference_sha, "checks": []}
        oracle_sha = json_file(self.oracle_path, oracle)
        contract = self.rows_contract()
        contract.update({
            "represented_input": {"path": "../inputs/represented.bin", "sha256": self.input_sha},
            "reference_output": {"path": "../inputs/rows-reference.bin", "sha256": reference_sha},
            "cpu_oracle": {"path": "../oracle/case.json", "sha256": oracle_sha},
        })
        contract_path = self.base / "contracts" / "rows-contract.json"
        json_file(contract_path, contract)
        resolved, resolved_sha = CHECK._validate_rows_contract_artifacts(
            self.base, contract_path, contract, self.input_path, self.input_sha,
            self.oracle_path, oracle_sha, oracle,
        )
        self.assertEqual(resolved, reference_path.resolve())
        self.assertEqual(resolved_sha, reference_sha)

        wrong_path = json.loads(json.dumps(contract))
        wrong_path["reference_output"]["path"] = "../inputs/not-the-reference.bin"
        with self.assertRaisesRegex(CHECK.EvidenceError, "missing or outside"):
            CHECK._validate_rows_contract_artifacts(
                self.base, contract_path, wrong_path, self.input_path, self.input_sha,
                self.oracle_path, oracle_sha, oracle,
            )

    def test_rows_controller_argv_is_bound_for_each_frozen_role(self):
        contract = self.rows_contract()
        reference_path = self.base / "inputs" / "rows-reference.bin"
        reference_sha = put(reference_path, b"row reference bytes")
        summaries = self.rows_role_summaries(contract, self.input_path, reference_path, reference_sha)
        implementations = {
            name: {"executed_path": entry["binary_path"]}
            for name, entry in self.impl.items()
        }
        CHECK._validate_rows_primary_argv(
            summaries, self.root, contract, self.input_path, self.input_sha,
            reference_path, reference_sha, implementations,
        )

        invalid_argvs = [
            ("sanitizer_candidate", lambda a: self.replace_argv_option(a, "--impl", "baseline")),
            ("candidate_timing", lambda a: self.replace_argv_option(a, "--case", "softmax_small")),
            ("profile_candidate", lambda a: self.replace_argv_option(a, "--repetitions", "21")),
            ("correctness_candidate", lambda a: a + ["--input", str(self.input_path.relative_to(self.root))]),
            ("correctness_baseline", lambda a: a + ["--unrecorded-option"]),
        ]
        for role, mutate in invalid_argvs:
            changed = json.loads(json.dumps(summaries))
            changed[role]["provenance"]["argv"] = mutate(changed[role]["provenance"]["argv"])
            with self.subTest(role=role), self.assertRaises(CHECK.EvidenceError):
                CHECK._validate_rows_primary_argv(
                    changed, self.root, contract, self.input_path, self.input_sha,
                    reference_path, reference_sha, implementations,
                )

        changed = json.loads(json.dumps(summaries))
        changed["sanitizer_candidate"]["provenance"]["argv"][0] = "/tmp/unrelated-row-runner"
        with self.assertRaisesRegex(CHECK.EvidenceError, r"argv\[0\] differs"):
            CHECK._validate_rows_primary_argv(
                changed, self.root, contract, self.input_path, self.input_sha,
                reference_path, reference_sha, implementations,
            )

        changed = json.loads(json.dumps(summaries))
        changed["baseline_timing"]["provenance"]["inputs"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(CHECK.EvidenceError, "do not bind the rows reference"):
            CHECK._validate_rows_primary_argv(
                changed, self.root, contract, self.input_path, self.input_sha,
                reference_path, reference_sha, implementations,
            )

    def test_rows_native_timing_keeps_unprofiled_samples_separate_from_profile(self):
        contract = self.rows_contract()
        for role in ("baseline_timing", "candidate_timing", "profile_candidate"):
            result = self.rows_timing_result(contract, role)
            CHECK._validate_rows_timing_output(result, contract, self.gpu, role)
        result = self.rows_timing_result(contract, "candidate_timing")
        result["cases"][0]["samples_us"] = [2.0]
        with self.assertRaisesRegex(CHECK.EvidenceError, "timing output is invalid"):
            CHECK._validate_rows_timing_output(result, contract, self.gpu, "candidate_timing")
        result = self.rows_timing_result(contract, "profile_candidate")
        result["cases"][0]["case_id"] = "rmsnorm_vec4_wide_streaming"
        with self.assertRaisesRegex(CHECK.EvidenceError, "timing output is invalid"):
            CHECK._validate_rows_timing_output(result, contract, self.gpu, "profile_candidate")

    def test_profile_report_must_be_present(self):
        self.manifest["cases"][0]["runs"]["profile_candidate"]["profile"].pop("report")
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("NCU report", result["errors"][0])

    def test_ncu_collection_shim_injects_only_the_approved_policy(self):
        fake_ncu = self.root / "fake-ncu"
        fake_ncu.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
        fake_ncu.chmod(0o755)
        log = self.base / "shim" / "argv.jsonl"
        env = {**os.environ, "NCU_REAL_BIN": str(fake_ncu), "NCU_SHIM_LOG": str(log)}
        result = subprocess.run([sys.executable, str(SHIM), "--target-processes", "all"],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), ["--import-source", "yes", "--clock-control", "none",
                                                    "--target-processes", "all"])
        record = json.loads(log.read_text().splitlines()[0])
        self.assertEqual(record["argv_received"], ["--target-processes", "all"])
        self.assertEqual(record["argv_executed"], json.loads(result.stdout))

    def test_ncu_query_calls_pass_through(self):
        fake_ncu = self.root / "fake-ncu"
        fake_ncu.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
        fake_ncu.chmod(0o755)
        env = {**os.environ, "NCU_REAL_BIN": str(fake_ncu), "NCU_SHIM_LOG": str(self.base / "query.jsonl")}
        result = subprocess.run([sys.executable, str(SHIM), "--query-metrics", "--chip", "sm_70"],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), ["--query-metrics", "--chip", "sm_70"])
        imported = subprocess.run([sys.executable, str(SHIM), "--import=selected.ncu-rep"],
                                  env=env, capture_output=True, text=True)
        self.assertEqual(imported.returncode, 0, imported.stderr)
        self.assertEqual(json.loads(imported.stdout), ["--import=selected.ncu-rep"])

    def test_ncu_shim_rejects_clock_changes(self):
        fake_ncu = self.root / "fake-ncu"
        fake_ncu.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
        fake_ncu.chmod(0o755)
        env = {**os.environ, "NCU_REAL_BIN": str(fake_ncu)}
        result = subprocess.run([sys.executable, str(SHIM), "--clock-control", "base"],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 64)
        self.assertIn("clock changes are disallowed", result.stderr)

    def test_ncu_shim_rejects_duplicate_policy_flags_in_both_forms(self):
        fake_ncu = self.root / "fake-ncu"
        fake_ncu.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
        fake_ncu.chmod(0o755)
        env = {**os.environ, "NCU_REAL_BIN": str(fake_ncu)}
        invocations = (
            (["--clock-control", "none", "--clock-control", "base"], "duplicate"),
            (["--clock-control=none", "--clock-control=base"], "duplicate"),
            (["--import-source", "yes", "--clock-control", "base"], "clock changes are disallowed"),
            (["--import-source=yes", "--import-source", "no"], "duplicate"),
        )
        for argv, expected_error in invocations:
            with self.subTest(argv=argv):
                result = subprocess.run([sys.executable, str(SHIM), *argv], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 64)
                self.assertIn(expected_error, result.stderr)

    def test_profile_receipt_rejects_changed_clock_policy(self):
        shim_log = self.base / self.runs["profile_candidate"]["profile"]["shim_log"]["path"]
        record = json.loads(shim_log.read_text())
        record["argv_executed"] = ["--clock-control", "base", "--import-source", "yes"]
        self.manifest["cases"][0]["runs"]["profile_candidate"]["profile"]["shim_log"]["sha256"] = json_file(shim_log, record).replace("\n", "")
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("no-clock/source-import policy", result["errors"][0])

    def test_profile_receipt_rejects_duplicate_clock_or_import_flags(self):
        shim_log = self.base / self.runs["profile_candidate"]["profile"]["shim_log"]["path"]
        record = json.loads(shim_log.read_text())
        record["argv_executed"] = ["--clock-control=none", "--clock-control", "base",
                                   "--import-source=yes", "--import-source", "no"]
        self.manifest["cases"][0]["runs"]["profile_candidate"]["profile"]["shim_log"]["sha256"] = json_file(
            shim_log, record)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("no-clock/source-import policy", result["errors"][0])

    def test_profile_receipt_accepts_single_equals_form_policy_flags(self):
        shim_log = self.base / self.runs["profile_candidate"]["profile"]["shim_log"]["path"]
        record = json.loads(shim_log.read_text())
        record["argv_executed"] = ["--clock-control=none", "--import-source=yes"]
        self.manifest["cases"][0]["runs"]["profile_candidate"]["profile"]["shim_log"]["sha256"] = json_file(
            shim_log, record)
        self.write_manifest()
        result = self.validate()
        self.assertTrue(result["valid"], result["errors"])

    def test_profile_receipt_rejects_any_unbound_collection_invocation(self):
        shim_log = self.base / self.runs["profile_candidate"]["profile"]["shim_log"]["path"]
        record = json.loads(shim_log.read_text())
        invalid = dict(record)
        invalid["argv_executed"] = ["--clock-control", "base", "--import-source", "yes"]
        self.manifest["cases"][0]["runs"]["profile_candidate"]["profile"]["shim_log"]["sha256"] = put(
            shim_log, (json.dumps(record, sort_keys=True) + "\n" +
                       json.dumps(invalid, sort_keys=True) + "\n").encode())
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("no-clock/source-import policy", result["errors"][0])

    def test_profile_receipt_binds_checked_in_shim_and_actual_ncu(self):
        shim_log = self.base / self.runs["profile_candidate"]["profile"]["shim_log"]["path"]
        record = json.loads(shim_log.read_text())
        record["shim_path"] = "/tmp/unrelated-shim.py"
        record["shim_sha256"] = "0" * 64
        self.manifest["cases"][0]["runs"]["profile_candidate"]["profile"]["shim_log"]["sha256"] = json_file(
            shim_log, record)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("no-clock/source-import policy", result["errors"][0])

        record["shim_path"] = str(self.repo_shim)
        record["shim_sha256"] = self.repo_shim_sha
        record["actual_ncu_path"] = "/tmp/unrelated-ncu"
        record["actual_ncu_sha256"] = "0" * 64
        self.manifest["cases"][0]["runs"]["profile_candidate"]["profile"]["shim_log"]["sha256"] = json_file(
            shim_log, record)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("no-clock/source-import policy", result["errors"][0])

    def test_profile_controller_inputs_bind_shim_and_actual_ncu_hashes(self):
        evidence = self.base / self.runs["profile_candidate"]["evidence"]["path"]
        exported = json.loads(evidence.read_text())
        inputs = exported["evidence"]["summary"]["provenance"]["inputs"]
        exported["evidence"]["summary"]["provenance"]["inputs"] = [
            item for item in inputs if item["path"] != str(self.fake_ncu.resolve())
        ]
        self.manifest["cases"][0]["runs"]["profile_candidate"]["evidence"]["sha256"] = json_file(
            evidence, exported)
        self.write_manifest()
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("no-clock/source-import policy", result["errors"][0])


class AttentionNativeEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((Path("/home/tumlinson/KernelWiki") / "benchmark_runs" /
                                   "volta-attention" / "cases" / "attention" /
                                   "attention_contract.json").read_text())
        cls.dimensions = CHECK._attention_case_dimensions(cls.contract)
        cls.oracle_data = {
            case_id: {
                "output_fp64": [0.0] * spec["outputs"],
                "rows": [{"valid_keys": 1, "normalizer": 1.0}] * spec["rows"],
            }
            for case_id, spec in cls.dimensions.items()
        }
        cls.gpu = "GPU-21131915-1488-23af-38dd-1743ae1f5cc8"

    def native_result(self, case="all", mode="correctness", implementation="online", splits=1):
        selected = list(self.dimensions) if case == "all" else [case]
        checks = []
        for case_id in selected:
            spec = self.dimensions[case_id]
            check = {
                "case": case_id, "passed": True,
                "checked_outputs": spec["outputs"], "checked_rows": spec["rows"],
                "max_abs_output_error": 0.0, "max_row_max_abs_error": 0.0,
                "max_row_normalizer_relative_error": 0.0,
            }
            if mode == "time":
                check.update({
                    "samples_us": [1.0] * 11,
                    "timing_phase": "steady_state_compute",
                    "timing_scope": {
                        "included": "full selected implementation",
                        "excluded": "input loading, device allocation, and H2D upload",
                    },
                })
            checks.append(check)
        errors = {
            "checked_outputs": sum(item["checked_outputs"] for item in checks),
            "checked_rows": sum(item["checked_rows"] for item in checks),
            "max_abs_output_error": 0.0,
            "max_row_max_abs_error": 0.0,
            "max_row_normalizer_relative_error": 0.0,
        }
        return {
            "format": CHECK.ATTENTION_RUN_FORMAT, "status": "passed",
            "compute_capability": "7.0", "gpu_uuid": "21131915148823af38dd1743ae1f5cc8",
            "case_id": "attention", "case": case, "mode": mode,
            "implementation": implementation, "splits": splits,
            "checks": checks, "errors": errors, "failure": None,
        }

    def validate_native(self, result, case="all", mode="correctness", implementation="online", splits=1):
        CHECK._attention_native_checks(result, self.contract, self.oracle_data,
                                       self.gpu, implementation, mode, case, splits,
                                       "attention fixture")

    def test_seven_case_attention_result_binds_all_output_and_row_counts(self):
        result = self.native_result()
        self.validate_native(result)
        self.assertEqual(result["errors"]["checked_outputs"], 39472)
        self.assertEqual(result["errors"]["checked_rows"], 552)

    def test_native_attention_rejects_wrong_case_counts_or_role_identity(self):
        result = self.native_result()
        result["checks"][0]["checked_outputs"] -= 1
        with self.assertRaisesRegex(CHECK.EvidenceError, "checked_outputs differs"):
            self.validate_native(result)

        result = self.native_result()
        result["implementation"] = "baseline"
        with self.assertRaisesRegex(CHECK.EvidenceError, "identity or status"):
            self.validate_native(result)

    def test_native_attention_rejects_errors_outside_frozen_tolerances(self):
        result = self.native_result()
        result["checks"][0]["max_row_max_abs_error"] = 0.001
        result["errors"]["max_row_max_abs_error"] = 0.001
        with self.assertRaisesRegex(CHECK.EvidenceError, "row maximum exceeds tolerance"):
            self.validate_native(result)

        result = self.native_result()
        result["checks"][0]["max_abs_output_error"] = 0.001
        result["errors"]["max_abs_output_error"] = 0.001
        with self.assertRaisesRegex(CHECK.EvidenceError, "output error exceeds"):
            self.validate_native(result)

    def test_attention_timing_requires_eleven_finite_scoped_samples(self):
        result = self.native_result("prefill64-long-causal", mode="time", implementation="baseline")
        self.validate_native(result, "prefill64-long-causal", "time", "baseline")
        result["checks"][0]["samples_us"][-1] = float("nan")
        with self.assertRaisesRegex(CHECK.EvidenceError, "timing samples or scope"):
            self.validate_native(result, "prefill64-long-causal", "time", "baseline")

    def test_attention_argv_rejects_duplicate_role_selectors(self):
        with self.assertRaisesRegex(CHECK.EvidenceError, "must contain 1 --impl selector"):
            CHECK._attention_selector(["--impl", "online", "--impl", "split"],
                                      "profile", "--impl", "online")

    def make_archive_capture(self, role, native_relative, outer_text, nested=False, portable=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            base = root / "benchmark_runs" / "attention"
            role_dir = base / "evidence" / "roles" / role
            role_dir.mkdir(parents=True)
            (root / ".git").mkdir(parents=True)
            lease_path = (base / "portable-leases" / f"{role}.json" if portable else
                          root / ".todo-orchestrator" / "runtime" / "foreground" / "lease.json")
            lease_sha = json_file(lease_path, {"format": "CUDA-FOREGROUND-LEASE/1"})

            evidence_id = "evidence-" + role
            execution_path = role_dir / "controller-run.json"
            execution_sha = json_file(execution_path, {"evidence_id": evidence_id})
            outer_archive_name = "controller.stdout.txt" if nested else "runner.stdout.txt"
            outer_path = role_dir / outer_archive_name
            outer_sha = put(outer_path, outer_text.encode())
            native_path = role_dir / native_relative
            native_sha = put(native_path, b'{"native":"attention"}\n')
            runtime_outer = f"/runtime/{role}/controller.stdout.txt"
            evidence_path = role_dir / "project-control-result.json"
            evidence = {
                "id": evidence_id,
                "artifacts": [{"kind": "stdout", "complete": 1,
                               "path": runtime_outer, "content_hash": outer_sha}],
                "summary": {"record": {"stdout": runtime_outer}},
            }
            evidence_sha = json_file(evidence_path, {"evidence": evidence})

            manifest_files = {
                "controller-run.json": execution_sha,
                "project-control-result.json": evidence_sha,
                "lease.json": lease_sha,
                outer_archive_name: outer_sha,
                native_relative: native_sha,
            }
            run = {
                "execution": {"path": str(execution_path.relative_to(base)), "sha256": execution_sha},
                "evidence": {"path": str(evidence_path.relative_to(base)), "sha256": evidence_sha},
                "lease": {
                    "path": str(lease_path.relative_to(base if portable else root)),
                    "sha256": lease_sha,
                    **({"original_path": "/retired/runtime/foreground/lease.json"}
                       if portable else {}),
                },
                "native_stdout_capture": {
                    "path": str(native_path.relative_to(base)), "sha256": native_sha,
                    **({"reported_path": "/runtime/inner/stdout.txt"} if nested else {}),
                },
            }
            sanitizer_summary = sanitizer_log = None
            if role.startswith("sanitizer_"):
                summary_path = role_dir / "compute-sanitizer" / "summary.json"
                summary_sha = json_file(summary_path, {"status": "partial"})
                log_path = role_dir / "compute-sanitizer" / "raw.log"
                log_sha = put(log_path, b"ERROR SUMMARY: 0 errors\n")
                manifest_files.update({
                    "compute-sanitizer/summary.json": summary_sha,
                    "compute-sanitizer/raw.log": log_sha,
                })
                sanitizer_summary = {"path": str(summary_path.relative_to(base)),
                                     "sha256": summary_sha}
                sanitizer_log = {"path": str(log_path.relative_to(base)),
                                 "sha256": log_sha,
                                 "reported_path": "/runtime/inner/raw.log"}
                run["sanitizer_summary"] = sanitizer_summary
                run["sanitizer_log"] = sanitizer_log
            manifest_path = role_dir / "archive-manifest.json"
            manifest_sha = json_file(manifest_path, {
                "format": "KERNELWIKI-ATTENTION-ROLE-ARCHIVE/1",
                "role": role, "evidence_id": evidence_id, "files": manifest_files,
            })
            run["archive_manifest"] = {"path": str(manifest_path.relative_to(base)),
                                        "sha256": manifest_sha}
            run["native_stdout"] = run["native_stdout_capture"]
            try:
                CHECK._attention_bind_stdout_capture(
                    base, root, run, role, native_path,
                    summary_path if role.startswith("sanitizer_") else None,
                    log_path if role.startswith("sanitizer_") else None,
                        portable=portable,
                )
            finally:
                self.assertTrue(root.exists())
            return run

    def test_attention_direct_stdout_is_bound_to_exported_controller_artifact(self):
        run = self.make_archive_capture(
            "candidate_decode_timing", "runner.stdout.txt", '{"native":"attention"}\n')
        self.assertEqual(run["native_stdout_capture"]["sha256"],
                         run["native_stdout"]["sha256"])

    def test_attention_nested_sanitizer_stdout_is_archived_separately(self):
        self.make_archive_capture(
            "sanitizer_candidate", "compute-sanitizer/stdout.txt",
            "wrapper summary\n- stdout: /runtime/inner/stdout.txt\n"
            "- raw log: /runtime/inner/raw.log\n", nested=True)

    def test_attention_portable_nested_stdout_uses_archive_without_runtime_tree(self):
        run = self.make_archive_capture(
            "sanitizer_candidate", "compute-sanitizer/stdout.txt",
            "wrapper summary\n- stdout: /runtime/inner/stdout.txt\n"
            "- raw log: /runtime/inner/raw.log\n", nested=True, portable=True)
        self.assertEqual(run["lease"]["original_path"],
                         "/retired/runtime/foreground/lease.json")


class GroupedLowbitNativeEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path("/home/tumlinson/KernelWiki")
        cls.bundle = cls.root / "benchmark_runs" / "volta-grouped-lowbit" / "cases" / "fixture_bundle"
        cls.gpu = "GPU-21131915-1488-23af-38dd-1743ae1f5cc8"
        cls.records = {}
        for suite, case_id in (("grouped", "kwv07-grouped-gated"), ("w4a16", "kwv07-w4a16")):
            contract_path = cls.bundle / "contracts" / f"{suite}-suite.json"
            input_path = cls.bundle / "bundle-manifest.json"
            oracle_path = cls.bundle / "oracles" / f"{suite}-suite.json"
            contract = CHECK.read_json(contract_path)
            cls.records[case_id] = CHECK._kwv07_suite_records(
                cls.root, case_id, contract_path, contract, input_path,
                CHECK.digest(input_path), oracle_path, CHECK.digest(oracle_path))

    def native_result(self, case_id, role):
        role_spec = CHECK.KWV07_ROLE_SPECS[case_id]["roles"][role]
        _impl_group, implementation, _kind, mode, selected = role_spec
        dimensions = self.records[case_id]["dimensions"]
        case_ids = list(dimensions) if selected is None else [selected]
        observed = []
        for case_name in case_ids:
            expected = dimensions[case_name]
            setup = 2.0
            predecode_event = 0.25 if implementation == "predecode-w4" else 0.0
            predecode_host = 0.5 if implementation == "predecode-w4" else 0.0
            item = {
                "case_id": case_name,
                "checked_outputs": expected["outputs"], "failed_outputs": 0,
                "nonfinite_outputs": 0, "max_abs_error": 0.0,
                "max_admission_tolerance": expected["atol"], "max_tolerance_ratio": 0.0,
                "one_time_setup_host_us": setup, "group_metadata_host_us": 0.5 if case_id == "kwv07-grouped-gated" else 0.0,
                "predecode_event_us": predecode_event, "predecode_host_us": predecode_host,
            }
            if mode == "time":
                events = [10.0] * 9
                hosts = [12.0] * 9
                item.update({"event_samples_us": events, "host_samples_us": hosts,
                             "median_event_us": 10.0, "median_host_us": 12.0})
                if case_id == "kwv07-grouped-gated":
                    item["cold_one_use_host_total_us"] = setup + 12.0
                else:
                    reuse = []
                    for uses in expected["reuse_cases"]:
                        row_events = [20.0] * 9
                        row_hosts = [22.0] * 9
                        reuse.append({
                            "uses": uses, "device_event_samples_us": row_events,
                            "compute_host_samples_us": row_hosts,
                            "cold_device_event_total_us": predecode_event + 20.0,
                            "cold_host_wall_total_us": setup + predecode_host + 22.0,
                        })
                    item["reuse_runs"] = reuse
                    item["reuse_assumption"] = CHECK.KWV07_REUSE_ASSUMPTION
                    item["event_samples_us"] = reuse[0]["device_event_samples_us"]
                    item["host_samples_us"] = reuse[0]["compute_host_samples_us"]
                    item["median_event_us"] = 20.0
                    item["median_host_us"] = 22.0
            observed.append(item)
        return {
            "format": CHECK.KWV07_NATIVE_FORMAT, "status": "passed",
            "suite": CHECK.KWV07_ROLE_SPECS[case_id]["suite"],
            "implementation": implementation, "mode": mode, "gpu": "NVIDIA V100",
            "gpu_uuid": "21131915148823af38dd1743ae1f5cc8",
            "compute_capability": "7.0", "case_count": len(case_ids),
            "checked_outputs": sum(item["checked_outputs"] for item in observed),
            "failed_outputs": 0, "nonfinite_outputs": 0,
            "max_abs_error": 0.0,
            "max_admission_tolerance": max(item["max_admission_tolerance"] for item in observed),
            "max_tolerance_ratio": 0.0, "cases": observed,
        }

    def test_both_native_suites_bind_exact_fp64_case_order_and_counts(self):
        self.assertEqual(self.records["kwv07-grouped-gated"]["total_outputs"], 70144)
        self.assertEqual(self.records["kwv07-w4a16"]["total_outputs"], 18479)
        self.assertEqual(len(self.records["kwv07-grouped-gated"]["dimensions"]), 6)
        self.assertEqual(len(self.records["kwv07-w4a16"]["dimensions"]), 4)
        self.assertTrue(all(record["atol"] > 0 for suite in self.records.values()
                            for record in suite["dimensions"].values()))

    def test_timing_scope_is_an_exact_machine_checked_declaration(self):
        for case_id, scope in CHECK.KWV07_TIMING_SCOPES.items():
            CHECK._kwv07_validate_timing_scope(case_id, {"timing_scope": scope})
            altered = json.loads(json.dumps(scope))
            altered["components"][next(iter(altered["components"]))] += " (claim added)"
            with self.assertRaisesRegex(CHECK.EvidenceError, "timing scope declaration"):
                CHECK._kwv07_validate_timing_scope(case_id, {"timing_scope": altered})

        grouped = CHECK.KWV07_TIMING_SCOPES["kwv07-grouped-gated"]
        self.assertIn("outside the CUDA-event scope", grouped["device_event"])
        self.assertIn("prep_us", grouped["components"]["group_metadata"])
        self.assertIn("one_time_setup_host_us", grouped["components"]["group_metadata"])

        w4 = CHECK.KWV07_TIMING_SCOPES["kwv07-w4a16"]["components"]
        self.assertIn("nine-sample medians", w4["predecode"])
        self.assertIn("derived subtotal", w4["derived_setup_prep_plus_batch_host_total"])
        self.assertIn("not full cold invocation latency",
                      w4["derived_setup_prep_plus_batch_host_total"])
        self.assertNotIn("cold_host_wall_total", w4)

    def test_native_adapter_accepts_each_declared_role_scope(self):
        for case_id, spec in CHECK.KWV07_ROLE_SPECS.items():
            for role in spec["roles"]:
                with self.subTest(case=case_id, role=role):
                    CHECK._kwv07_native_checks(self.native_result(case_id, role), case_id,
                                               self.records[case_id], self.gpu, role, role)

    def test_archived_spec_adapters_bind_exact_runner_selectors_and_build_once(self):
        binary = self.root / "benchmark_runs" / "volta-grouped-lowbit" / "bin" / "v1" / "kwv07_runner"
        for case_id, family in CHECK.KWV07_ROLE_SPECS.items():
            for role, expected in family["roles"].items():
                with self.subTest(case=case_id, role=role):
                    spec_path = self.root / "benchmark_runs" / "volta-grouped-lowbit" / "controller" / f"role-{role}.spec.json"
                    spec = CHECK.read_json(spec_path)
                    CHECK._kwv07_validate_spec(
                        self.root, case_id, role, {"spec": spec}, binary, self.gpu, expected)
                    argv = [str(binary), "--bundle",
                            "benchmark_runs/volta-grouped-lowbit/cases/fixture_bundle",
                            "--suite", family["suite"], "--impl", expected[1],
                            "--mode", expected[3]]
                    if expected[4] is not None:
                        argv += ["--case", expected[4]]
                    CHECK._kwv07_validate_argv(
                        {"provenance": {"argv": argv}}, self.root, case_id, role,
                        {"executed_path": binary}, expected)
                    if role != "grouped_correctness_baseline":
                        tampered = json.loads(json.dumps(spec))
                        tampered.setdefault("benchmark", {})["build_argv"] = ["nvcc"]
                        with self.assertRaisesRegex(CHECK.EvidenceError, "rebuilt after the first role"):
                            CHECK._kwv07_validate_spec(
                                self.root, case_id, role, {"spec": tampered}, binary,
                                self.gpu, expected)
        case_id, role = "kwv07-grouped-gated", "grouped_profile_candidate"
        expected = CHECK.KWV07_ROLE_SPECS[case_id]["roles"][role]
        argv = [str(binary), "--bundle", "benchmark_runs/volta-grouped-lowbit/cases/fixture_bundle",
                "--suite", "grouped", "--impl", "paired-gated", "--mode", "check",
                "--case", "weighted_skewed", "--warmups", "2"]
        with self.assertRaisesRegex(CHECK.EvidenceError, "selectors differ"):
            CHECK._kwv07_validate_argv(
                {"provenance": {"argv": argv}}, self.root, case_id, role,
                {"executed_path": binary}, expected)

    def test_native_adapter_rejects_case_count_policy_and_identity_changes(self):
        case_id, role = "kwv07-grouped-gated", "grouped_correctness_candidate"
        result = self.native_result(case_id, role)
        result["cases"].pop()
        with self.assertRaisesRegex(CHECK.EvidenceError, "case IDs/order"):
            CHECK._kwv07_native_checks(result, case_id, self.records[case_id], self.gpu, role, role)

        result = self.native_result(case_id, role)
        result["gpu_uuid"] = "f" * 32
        with self.assertRaisesRegex(CHECK.EvidenceError, "identity, GPU"):
            CHECK._kwv07_native_checks(result, case_id, self.records[case_id], self.gpu, role, role)

        result = self.native_result(case_id, role)
        result["cases"][0]["max_admission_tolerance"] *= 2
        result["max_admission_tolerance"] *= 2
        with self.assertRaisesRegex(CHECK.EvidenceError, "frozen per-output policy"):
            CHECK._kwv07_native_checks(result, case_id, self.records[case_id], self.gpu, role, role)

    def test_grouped_timing_requires_nine_unprofiled_samples_and_matching_total(self):
        case_id, role = "kwv07-grouped-gated", "grouped_candidate_timing"
        result = self.native_result(case_id, role)
        CHECK._kwv07_native_checks(result, case_id, self.records[case_id], self.gpu, role, role)
        result["cases"][0]["event_samples_us"].pop()
        with self.assertRaisesRegex(CHECK.EvidenceError, "nine positive finite samples"):
            CHECK._kwv07_native_checks(result, case_id, self.records[case_id], self.gpu, role, role)

        result = self.native_result(case_id, role)
        result["cases"][0]["cold_one_use_host_total_us"] += 1
        with self.assertRaisesRegex(CHECK.EvidenceError, "cold host total"):
            CHECK._kwv07_native_checks(result, case_id, self.records[case_id], self.gpu, role, role)

    def test_w4_reuse_timings_bind_each_contract_reuse_count_and_cold_totals(self):
        case_id, role = "kwv07-w4a16", "w4a16_baseline_timing"
        result = self.native_result(case_id, role)
        CHECK._kwv07_native_checks(result, case_id, self.records[case_id], self.gpu, role, role)
        row = result["cases"][0]["reuse_runs"][-1]
        row["cold_host_wall_total_us"] += 1
        with self.assertRaisesRegex(CHECK.EvidenceError, "cold total"):
            CHECK._kwv07_native_checks(result, case_id, self.records[case_id], self.gpu, role, role)

        result = self.native_result(case_id, role)
        result["cases"][0]["reuse_runs"][0]["uses"] = 2
        with self.assertRaisesRegex(CHECK.EvidenceError, "reuse protocol"):
            CHECK._kwv07_native_checks(result, case_id, self.records[case_id], self.gpu, role, role)

    def test_ncu_check_profile_is_single_selected_case_without_timing_data(self):
        case_id, role = "kwv07-w4a16", "w4a16_profile_candidate"
        result = self.native_result(case_id, role)
        self.assertEqual(result["mode"], "check")
        self.assertEqual([item["case_id"] for item in result["cases"]], ["weighted_low_m"])
        CHECK._kwv07_native_checks(result, case_id, self.records[case_id], self.gpu, role, role)
        result["cases"][0]["event_samples_us"] = [1.0] * 9
        with self.assertRaisesRegex(CHECK.EvidenceError, "contains timing measurements"):
            CHECK._kwv07_native_checks(result, case_id, self.records[case_id], self.gpu, role, role)

    def make_source_closure(self, root):
        base = root / "benchmark_runs" / "volta-grouped-lowbit"
        archive_root = base / "evidence" / "v1" / "source-closure" / "repo"
        roles = list(CHECK.KWV07_ROLE_SPECS["kwv07-grouped-gated"]["roles"])
        roles += list(CHECK.KWV07_ROLE_SPECS["kwv07-w4a16"]["roles"])
        spec_paths = [f"benchmark_runs/volta-grouped-lowbit/controller/role-{role}.spec.json"
                      for role in roles]
        source_path = "artifacts/kernels/volta-grouped-lowbit/variants/grouped_lowbit.cu"
        all_inputs = [source_path, *spec_paths]
        files = []
        captured = []
        declaring = []
        for path in [source_path, *spec_paths]:
            archive_path = archive_root / f"input-{len(files):02}.bin"
            if path in spec_paths:
                role = roles[spec_paths.index(path)]
                sha = json_file(archive_path, {"role": role, "contract_id":
                                               ("kwv07-grouped-gated" if role.startswith("grouped_")
                                                else "kwv07-w4a16"),
                                               "inputs": all_inputs})
                declaring.append({"path": path, "sha256": sha})
            else:
                sha = put(archive_path, b"grouped and W4 shared source\n")
            files.append({"original_path": path,
                          "archived_path": str(archive_path.relative_to(root)),
                          "sha256": sha, "size_bytes": archive_path.stat().st_size})
            captured.append({"original_path": path,
                             "archive_path": str(archive_path.relative_to(root)),
                             "sha256": sha})
        closure_path = base / "evidence" / "v1" / "source-closure" / "source-closure.json"
        closure_data = {"format": "KERNELWIKI-KWV07-SOURCE-CLOSURE/1",
                        "repository_inputs": files, "declaring_specs": declaring,
                        "tools": []}
        closure_sha = json_file(closure_path, closure_data)
        return ({"source_closure": {"path": str(closure_path.relative_to(root)),
                                     "sha256": closure_sha},
                 "captured_input_archives": captured}, closure_path)

    def test_source_closure_requires_all_twelve_role_specs_and_exact_archives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            case, _closure_path = self.make_source_closure(root)
            closure, captures = CHECK._kwv07_source_closure(
                root, case, "kwv07-grouped-gated")
            self.assertEqual(len(closure["declared_specs"]), 12)
            self.assertEqual(len(captures), 13)
            case["captured_input_archives"].pop()
            with self.assertRaisesRegex(CHECK.EvidenceError, "aliases differ"):
                CHECK._kwv07_source_closure(root, case, "kwv07-grouped-gated")

    def make_kwv07_stdout_capture(self, root, role, nested=False, sanitizer=False):
        base = root / "benchmark_runs" / "volta-grouped-lowbit"
        role_dir = base / "evidence" / "v1" / "roles" / role
        evidence_id = "kwv07-evidence-" + role
        execution_path = role_dir / "controller-run.json"
        execution_sha = json_file(execution_path, {"evidence_id": evidence_id})
        lease_path = root / ".todo-orchestrator" / "runtime" / "foreground" / "lease.json"
        lease_sha = json_file(lease_path, {"format": "CUDA-FOREGROUND-LEASE/1"})
        if sanitizer:
            native_relative = "sanitizer/run/stdout.txt"
            outer_text = ("wrapper summary\n- stdout: /runtime/kwv07/stdout.txt\n"
                          "- raw log: /runtime/kwv07/raw.log\n")
        elif nested:
            native_relative = "ncu/run/command.stdout.txt"
            outer_text = "profile summary\n- command stdout: /runtime/kwv07/ncu/run/command.stdout.txt\n"
        else:
            native_relative = "foreground.stdout.txt"
            outer_text = "{\"format\":\"KERNELWIKI-KWV07-NATIVE/1\"}\n"
        outer_path = role_dir / "foreground.stdout.txt"
        outer_sha = put(outer_path, outer_text.encode())
        native_path = role_dir / native_relative
        native_sha = put(native_path, b'{"format":"KERNELWIKI-KWV07-NATIVE/1"}\n')
        runtime_outer = f"/runtime/{role}/foreground.stdout.txt"
        evidence_path = role_dir / "project-control-result.json"
        evidence = {
            "id": evidence_id,
            "artifacts": [{"kind": "stdout", "complete": 1,
                           "path": runtime_outer, "content_hash": outer_sha}],
            "summary": {"record": {"stdout": runtime_outer}},
        }
        evidence_sha = json_file(evidence_path, {"evidence": evidence})
        files = {
            "controller-run.json": execution_sha,
            "project-control-result.json": evidence_sha,
            "lease.json": lease_sha,
            "foreground.stdout.txt": outer_sha,
            "foreground.stderr.txt": put(role_dir / "foreground.stderr.txt", b""),
            native_relative: native_sha,
        }
        run = {
            "execution": {"path": str(execution_path.relative_to(base)), "sha256": execution_sha},
            "evidence": {"path": str(evidence_path.relative_to(base)), "sha256": evidence_sha},
            "lease": {"path": str(lease_path.relative_to(root)), "sha256": lease_sha},
            "native_stdout_capture": {"path": str(native_path.relative_to(base)),
                                      "sha256": native_sha,
                                      **({"reported_path": "/runtime/kwv07/stdout.txt"} if sanitizer else
                                         {"reported_path": "/runtime/kwv07/ncu/run/command.stdout.txt"} if nested else {})},
        }
        sanitizer_summary = sanitizer_log = None
        if sanitizer:
            sanitizer_summary = role_dir / "sanitizer" / "run" / "summary.json"
            summary_sha = json_file(sanitizer_summary, {
                "tool": "compute-sanitizer", "status": "partial", "tool_detail": "memcheck",
                "conclusive": False, "exit_status": 0, "error_count": 0,
            })
            sanitizer_log = role_dir / "sanitizer" / "run" / "raw.log"
            log_sha = put(sanitizer_log, b"========= ERROR SUMMARY: 0 errors\n")
            files["sanitizer/run/summary.json"] = summary_sha
            files["sanitizer/run/raw.log"] = log_sha
            run["sanitizer_summary"] = {"path": str(sanitizer_summary.relative_to(base)),
                                        "sha256": summary_sha}
            run["sanitizer_log"] = {"path": str(sanitizer_log.relative_to(base)),
                                    "sha256": log_sha,
                                    "reported_path": "/runtime/kwv07/raw.log"}
        manifest_path = role_dir / "manifest.json"
        manifest_sha = json_file(manifest_path, {
            "role": role, "evidence_id": evidence_id,
            "gpu_uuid_raw_hex": "21131915148823af38dd1743ae1f5cc8",
            "compute_capability": "7.0", "files": files,
        })
        run["archive_manifest"] = {"path": str(manifest_path.relative_to(base)),
                                   "sha256": manifest_sha}
        return base, run, native_path, sanitizer_summary, sanitizer_log

    def test_controller_stdout_export_hash_binds_direct_runner_json(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            base, run, native_path, _, _ = self.make_kwv07_stdout_capture(
                root, "grouped_correctness_baseline")
            archive = CHECK._kwv07_bind_stdout_capture(
                base, root, "kwv07-grouped-gated", "grouped_correctness_baseline",
                self.gpu, run, native_path)
            self.assertEqual(archive["role"], "grouped_correctness_baseline")
            run["native_stdout_capture"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(CHECK.EvidenceError, "SHA-256 mismatch"):
                CHECK._kwv07_bind_stdout_capture(
                    base, root, "kwv07-grouped-gated", "grouped_correctness_baseline",
                    self.gpu, run, native_path)

    def test_nested_ncu_stdout_is_path_and_archive_bound_but_not_a_timing_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            base, run, native_path, _, _ = self.make_kwv07_stdout_capture(
                root, "grouped_profile_candidate", nested=True)
            CHECK._kwv07_bind_stdout_capture(
                base, root, "kwv07-grouped-gated", "grouped_profile_candidate",
                self.gpu, run, native_path)
            run["native_stdout_capture"]["reported_path"] = "/runtime/other.json"
            with self.assertRaisesRegex(CHECK.EvidenceError, "not identified by captured outer output"):
                CHECK._kwv07_bind_stdout_capture(
                    base, root, "kwv07-grouped-gated", "grouped_profile_candidate",
                    self.gpu, run, native_path)

    def test_ncu_profile_files_include_hash_bound_raw_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            base, run, _native_path, _, _ = self.make_kwv07_stdout_capture(
                root, "grouped_profile_candidate", nested=True)
            manifest_path = CHECK.checked_file(base, run["archive_manifest"], "manifest")
            manifest = CHECK.read_json(manifest_path)
            role_dir = manifest_path.parent
            profile = {}
            for key, relative in (("report", "ncu/report.ncu-rep"),
                                  ("analysis", "ncu/summary.json"),
                                  ("raw_csv", "ncu/raw.csv"),
                                  ("shim_log", "ncu-shim.log")):
                path = role_dir / relative
                if key == "analysis":
                    sha = json_file(path, {"counter_valid": True, "timing_valid": False})
                else:
                    sha = put(path, (key + " evidence\n").encode())
                manifest["files"][relative] = sha
                profile[key] = {"path": str(path.relative_to(base)), "sha256": sha}
            json_file(manifest_path, manifest)
            run["archive_manifest"]["sha256"] = CHECK.digest(manifest_path)
            run["profile"] = profile
            CHECK._kwv07_bind_profile_files(
                base, run, "profile", CHECK.read_json(manifest_path))
            profile["raw_csv"]["path"] = str((role_dir / "ncu" / "not-raw.csv").relative_to(base))
            (role_dir / "ncu" / "not-raw.csv").write_text("counter\n")
            sha = CHECK.digest(role_dir / "ncu" / "not-raw.csv")
            profile["raw_csv"]["sha256"] = sha
            manifest["files"]["ncu/not-raw.csv"] = sha
            json_file(manifest_path, manifest)
            run["archive_manifest"]["sha256"] = CHECK.digest(manifest_path)
            with self.assertRaisesRegex(CHECK.EvidenceError, "raw counter CSV path"):
                CHECK._kwv07_bind_profile_files(
                    base, run, "profile", CHECK.read_json(manifest_path))

    def test_sanitizer_raw_zero_partial_classifier_keeps_app_stdout_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            base, run, native_path, summary, raw_log = self.make_kwv07_stdout_capture(
                root, "grouped_sanitizer_candidate", sanitizer=True)
            CHECK._kwv07_bind_stdout_capture(
                base, root, "kwv07-grouped-gated", "grouped_sanitizer_candidate",
                self.gpu, run, native_path, summary, raw_log)
            raw_log.write_text("========= ERROR SUMMARY: 1 errors\n")
            run["sanitizer_log"]["sha256"] = CHECK.digest(raw_log)
            manifest_path = CHECK.checked_file(base, run["archive_manifest"], "manifest")
            manifest = CHECK.read_json(manifest_path)
            manifest["files"]["sanitizer/run/raw.log"] = CHECK.digest(raw_log)
            json_file(manifest_path, manifest)
            run["archive_manifest"]["sha256"] = CHECK.digest(manifest_path)
            with self.assertRaisesRegex(CHECK.EvidenceError, "exactly zero errors"):
                CHECK._kwv07_bind_stdout_capture(
                    base, root, "kwv07-grouped-gated", "grouped_sanitizer_candidate",
                    self.gpu, run, native_path, summary, raw_log)


class RecurrenceNativeEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path("/home/tumlinson/KernelWiki")
        cls.base = cls.root / "benchmark_runs" / "volta-recurrence"
        cls.case = CHECK.read_json(cls.base / "summary.template.json")["cases"][0]
        cls.contract = CHECK.read_json(cls.base / "contract.json")
        cls.oracle_path = cls.base / "cpu-oracle.json"
        cls.input_path = cls.base / "inputs" / "recurrence_inputs.bin"
        cls.records = CHECK._kwv08_suite_records(
            cls.base, cls.root, cls.case, cls.base / "contract.json", cls.contract,
            cls.input_path, CHECK.digest(cls.input_path), cls.oracle_path,
            CHECK.digest(cls.oracle_path))
        cls.gpu = "GPU-21131915-1488-23af-38dd-1743ae1f5cc8"

    def native_result(self, role):
        _kind, _implementation, variant, mode, selected = CHECK.KWV08_ROLES[role]
        case_ids = [selected] if selected else list(self.records["cases"])
        native_cases = []
        for case_id in case_ids:
            expected = self.records["cases"][case_id]
            checks = []
            names = list(CHECK.KWV08_CHECKS)
            if mode == "profile":
                names = names[:3]
            counts = {
                "outputs": expected["live_steps"] * expected["d_v"],
                "padding": (expected["T"] * expected["B"] - expected["live_steps"]) * expected["d_v"],
                "state": expected["B"] * expected["d_v"] * expected["d_k"],
            }
            for name in names:
                kind = CHECK.KWV08_CHECKS[name]
                checks.append({
                    "name": name, "status": "passed", "checked": counts[kind],
                    "failed": 0, "nonfinite": 0,
                    "max_abs_error": 0.0, "max_scaled_error": 0.0,
                })
            timing = {"status": "not_run"}
            if mode == "time":
                timing = {"status": "passed"}
                for field, median in (("host_enqueue_us", "host_enqueue_median_us"),
                                      ("host_completion_us", "host_completion_median_us"),
                                      ("device_event_interval_us", "device_event_interval_median_us")):
                    timing[field] = [float(value) for value in range(1, 8)]
                    timing[median] = 4.0
            native_cases.append({
                "case_id": case_id,
                "shape": {key: expected[key] for key in ("T", "B", "d_v", "d_k")},
                "variant": variant, "status": "passed", "checks": checks,
                "timing": timing,
            })
        return {
            "format": CHECK.KWV08_NATIVE_FORMAT, "status": "passed",
            "variant": variant, "mode": mode, "gpu": "Tesla V100-SXM2-32GB",
            "gpu_uuid": self.gpu.removeprefix("GPU-").replace("-", "").lower(),
            "compute_capability": "7.0",
            "bundle_file_sha256": self.records["bundle_file_sha256"],
            "bundle_payload_sha256": self.records["bundle_payload_sha256"],
            "tolerance": {"atol": self.records["atol"], "rtol": self.records["rtol"]},
            "timing_semantics": {
                "host_enqueue_us": "wall time through submission of all launches",
                "host_completion_us": "wall time through stream completion",
                "device_event_interval_us": "CUDA event interval; can include device idle while host submits per-token launches",
            },
            "cases": native_cases,
        }

    def test_cpu_contract_bundle_and_oracle_are_bound(self):
        self.assertEqual(list(self.records["cases"]), [
            "decode_many_sequences", "short_prefill_many_sequences", "long_prefill_many_sequences"])
        self.assertEqual(self.records["bundle_file_sha256"], CHECK.KWV08_INPUT_SHA256)
        self.assertEqual(self.records["bundle_payload_sha256"], CHECK.KWV08_PAYLOAD_SHA256)
        self.assertEqual(self.records["cases"]["long_prefill_many_sequences"]["live_steps"], 13014)

    def test_archived_source_closure_binds_six_specs_and_all_captured_inputs(self):
        closure_path = self.base / "evidence" / "v1" / "source-closure" / "source-closure.json"
        closure = CHECK.read_json(closure_path)
        case = {
            "id": "recurrence_state_ownership_bundle",
            "gpu_uuid": closure["gpu_uuid_target"],
            "source_closure": {
                "path": str(closure_path.relative_to(self.root)),
                "sha256": CHECK.digest(closure_path),
            },
            "captured_input_archives": [
                {"original_path": row["original_path"],
                 "archive_path": row["archived_path"], "sha256": row["sha256"]}
                for row in closure["repository_inputs"]
            ],
        }
        # The portable receipt keeps its frozen closure and archive hashes authoritative
        # without requiring the historical CUDA installation to be present on this host.
        info, aliases, toolchain = CHECK._kwv08_source_closure(
            self.root, case, portable=True)
        self.assertEqual(info["sha256"], CHECK.KWV08_SOURCE_CLOSURE_SHA256)
        self.assertEqual(len(aliases), 31)
        self.assertEqual(len(toolchain), 8)
        self.assertEqual({item["role"] for item in info["declared_specs"].values()},
                         {f"recurrence_{role}" for role in CHECK.KWV08_ROLES})

    def test_live_source_closure_rejects_toolchain_file_drift(self):
        closure_path = self.base / "evidence" / "v1" / "source-closure" / "source-closure.json"
        closure = CHECK.read_json(closure_path)
        case = {
            "id": "recurrence_state_ownership_bundle",
            "gpu_uuid": closure["gpu_uuid_target"],
            "source_closure": {
                "path": str(closure_path.relative_to(self.root)),
                "sha256": CHECK.digest(closure_path),
            },
            "captured_input_archives": [
                {"original_path": row["original_path"],
                 "archive_path": row["archived_path"], "sha256": row["sha256"]}
                for row in closure["repository_inputs"]
            ],
        }
        changed = closure["toolchain_inputs"][0]
        changed_path = changed["path"]
        original_digest = CHECK.digest
        original_is_file = CHECK.Path.is_file
        original_stat = CHECK.Path.stat
        changed_hash = "0" * 64 if changed["sha256"] != "0" * 64 else "1" * 64

        def is_file(path):
            return True if str(path) == changed_path else original_is_file(path)

        def stat(path, *args, **kwargs):
            if str(path) == changed_path:
                return SimpleNamespace(st_size=changed["size_bytes"])
            return original_stat(path, *args, **kwargs)

        def digest(path):
            return changed_hash if str(path) == changed_path else original_digest(path)

        with patch.object(CHECK.Path, "is_file", autospec=True, side_effect=is_file), \
                patch.object(CHECK.Path, "stat", autospec=True, side_effect=stat), \
                patch.object(CHECK, "digest", side_effect=digest):
            with self.assertRaisesRegex(
                    CHECK.EvidenceError, "current toolchain file differs from captured identity"):
                CHECK._kwv08_source_closure(self.root, case)

    def test_copied_file_paths_accept_collector_archive_aliases_without_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            role_dir = root / "benchmark_runs/volta-recurrence/evidence/v1/roles/sanitizer"
            copied = role_dir / "sanitizer/run/raw.log"
            put(copied, b"ERROR SUMMARY: 0 errors\n")
            expected = copied.resolve()
            self.assertEqual(
                CHECK._kwv08_copied_role_file(root, role_dir, str(copied), "absolute"),
                expected,
            )
            self.assertEqual(
                CHECK._kwv08_copied_role_file(
                    root, role_dir,
                    "benchmark_runs/volta-recurrence/evidence/v1/roles/sanitizer/sanitizer/run/raw.log",
                    "repository relative"),
                expected,
            )
            self.assertEqual(
                CHECK._kwv08_copied_role_file(
                    root, role_dir, "sanitizer/run/raw.log", "role relative"),
                expected,
            )
            outside = root / "outside.txt"
            put(outside, b"not an archived role artifact")
            with self.assertRaisesRegex(CHECK.EvidenceError, "escapes its role archive"):
                CHECK._kwv08_copied_role_file(root, role_dir, "../outside.txt", "escape")

    def test_captured_sanitizer_archive_binds_nested_app_and_path_report(self):
        role_dir = self.base / "evidence/v1/roles/recurrence_sanitizer_candidate"
        archive = role_dir / "archive-manifest.json"
        path_report = CHECK.read_json(role_dir / "sanitizer-path-report.json")
        archive_prefix = role_dir.relative_to(self.root).as_posix()

        def reported(relative):
            archive_path = f"{archive_prefix}/{relative}"
            return next(item["source_path"] for item in path_report["files"]
                        if item["archive_path"] == archive_path)

        def ref(relative):
            path = role_dir / relative
            return {"path": path.relative_to(self.base).as_posix(),
                    "sha256": CHECK.digest(path)}

        run = {
            "archive_manifest": ref("archive-manifest.json"),
            "evidence": ref("controller-evidence.json"),
            "execution": ref("controller-run.json"),
            "lease": {"path": role_dir.relative_to(self.root).joinpath("lease.json").as_posix(),
                      "sha256": CHECK.digest(role_dir / "lease.json")},
            "project_control_raw": ref("project-control-read-export.json"),
            "native_stdout_capture": {
                **ref("sanitizer/run/stdout.txt"),
                "reported_path": reported("sanitizer/run/stdout.txt"),
            },
            "sanitizer_summary": {
                **ref("sanitizer/run/summary.json"),
                "reported_path": reported("sanitizer/run/summary.json"),
            },
            "sanitizer_log": {
                **ref("sanitizer/run/raw.log"),
                "reported_path": reported("sanitizer/run/raw.log"),
            },
        }
        CHECK._kwv08_bind_stdout_capture(
            self.base, self.root, "sanitizer_candidate", self.gpu, run,
            role_dir / "sanitizer/run/stdout.txt")
        self.assertEqual(CHECK.read_json(role_dir / "sanitizer/run/summary.json")["status"],
                         "partial")

    def test_captured_ncu_archive_binds_selected_stdout_and_excludes_profile_timing(self):
        role_dir = self.base / "evidence/v1/roles/recurrence_profile_candidate"
        outer_text = (role_dir / "foreground.stdout.txt").read_text()
        reported_stdout = next(line.split("command stdout: ", 1)[1].strip()
                               for line in outer_text.splitlines()
                               if "command stdout: " in line)

        def ref(relative):
            path = role_dir / relative
            return {"path": path.relative_to(self.base).as_posix(),
                    "sha256": CHECK.digest(path)}

        run = {
            "archive_manifest": ref("archive-manifest.json"),
            "evidence": ref("controller-evidence.json"),
            "execution": ref("controller-run.json"),
            "lease": {"path": role_dir.relative_to(self.root).joinpath("lease.json").as_posix(),
                      "sha256": CHECK.digest(role_dir / "lease.json")},
            "project_control_raw": ref("project-control-read-export.json"),
            "native_stdout_capture": {
                **ref("ncu/run/command.stdout.txt"), "reported_path": reported_stdout,
            },
            "profile": {
                "summary": ref("ncu/run/summary.json"),
                "analysis": ref("ncu/run/analysis.txt"),
                "raw_csv": ref("ncu/run/raw.csv"),
                "report": ref("ncu/run/report.ncu-rep"),
                "shim_log": ref("ncu-shim.jsonl"),
            },
        }
        CHECK._kwv08_bind_stdout_capture(
            self.base, self.root, "profile_candidate", self.gpu, run,
            role_dir / "ncu/run/command.stdout.txt")
        native = CHECK.read_json(role_dir / "ncu/run/command.stdout.txt")
        CHECK._kwv08_native_checks(native, self.records, self.gpu,
                                   "profile_candidate", "profile candidate")
        self.assertFalse(CHECK.read_json(role_dir / "ncu/run/summary.json")["timing_valid"])

    def test_all_role_stdout_contracts_require_every_output_and_state(self):
        for role in CHECK.KWV08_ROLES:
            with self.subTest(role=role):
                CHECK._kwv08_native_checks(self.native_result(role), self.records,
                                           self.gpu, role, role)

    def test_missing_segmented_final_state_is_rejected(self):
        native = self.native_result("correctness_candidate")
        native["cases"][0]["checks"] = [
            item for item in native["cases"][0]["checks"]
            if item["name"] != "segmented_carry_final_state"]
        with self.assertRaisesRegex(CHECK.EvidenceError, "check set is incomplete"):
            CHECK._kwv08_native_checks(native, self.records, self.gpu,
                                      "correctness_candidate", "candidate")

    def test_output_extent_and_tolerance_ratio_are_enforced(self):
        native = self.native_result("correctness_baseline")
        native["cases"][1]["checks"][0]["checked"] -= 1
        with self.assertRaisesRegex(CHECK.EvidenceError, "full expected extent"):
            CHECK._kwv08_native_checks(native, self.records, self.gpu,
                                      "correctness_baseline", "baseline")
        native = self.native_result("correctness_baseline")
        native["cases"][0]["checks"][0]["max_scaled_error"] = 1.01
        with self.assertRaisesRegex(CHECK.EvidenceError, "exceeds frozen tolerance"):
            CHECK._kwv08_native_checks(native, self.records, self.gpu,
                                      "correctness_baseline", "baseline")

    def test_timing_medians_and_profile_scope_are_separate(self):
        native = self.native_result("candidate_timing")
        native["cases"][0]["timing"]["host_enqueue_median_us"] = 3.5
        with self.assertRaisesRegex(CHECK.EvidenceError, "differs from its samples"):
            CHECK._kwv08_native_checks(native, self.records, self.gpu,
                                      "candidate_timing", "candidate timing")
        profile = self.native_result("profile_candidate")
        profile["cases"][0]["timing"] = {
            "status": "passed", "device_event_interval_us": [42.0]}
        with self.assertRaisesRegex(CHECK.EvidenceError, "unexpectedly contains timing evidence"):
            CHECK._kwv08_native_checks(profile, self.records, self.gpu,
                                      "profile_candidate", "profile")


if __name__ == "__main__":
    unittest.main()
