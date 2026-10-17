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


if __name__ == "__main__":
    unittest.main()
