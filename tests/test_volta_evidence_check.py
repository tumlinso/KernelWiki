import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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
        cls.contract = json.loads((Path(__file__).parents[1] / "benchmark_runs" /
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

    def make_archive_capture(self, role, native_relative, outer_text, nested=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            base = root / "benchmark_runs" / "attention"
            role_dir = base / "evidence" / "roles" / role
            role_dir.mkdir(parents=True)
            (root / ".git").mkdir(parents=True)
            lease_path = root / ".todo-orchestrator" / "runtime" / "foreground" / "lease.json"
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
                "lease": {"path": str(lease_path.relative_to(root)), "sha256": lease_sha},
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


if __name__ == "__main__":
    unittest.main()
