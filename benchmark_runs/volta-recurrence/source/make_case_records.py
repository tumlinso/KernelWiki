#!/usr/bin/env python3
"""Create KWV-08 evidence contract and CPU-oracle records from one bundle."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from bundle_io import ATOL, RTOL, decode_bundle
from recurrence_oracle import run_sequence

CASE_ID = "recurrence_state_ownership_bundle"
BUNDLE_PATH = "benchmark_runs/volta-recurrence/inputs/recurrence_inputs.bin"
CONTRACT_PATH = Path("contract.json")
ORACLE_PATH = Path("cpu-oracle.json")
SUMMARY_TEMPLATE_PATH = Path("summary.template.json")


def canonical_sha(value) -> str:
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def segmented_check(case):
    t,batch,dv,dk=case.shape
    boundaries=sorted(set([0,min(1,t),min(33,t),min(129,t),t]))
    expected_parts=np.full((t,batch,dv),np.nan,dtype=np.float64)
    final=np.empty((batch,dv,dk),dtype=np.float64)
    for seq in range(batch):
        state=case.initial_states[seq].astype(np.float64)
        length=int(case.lengths[seq])
        for start,end in zip(boundaries,boundaries[1:]):
            live_end=min(end,length)
            if live_end <= start:
                continue
            out,state=run_sequence(
                case.keys[start:live_end,seq],case.values[start:live_end,seq],
                case.queries[start:live_end,seq],case.alpha[start:live_end,seq],
                case.beta[start:live_end,seq],state
            )
            expected_parts[start:live_end,seq]=out
        final[seq]=state
    live=np.arange(t)[:,None] < case.lengths[None,:]
    out_err=np.abs(expected_parts[live]-case.expected_outputs[live])
    final_err=np.abs(final-case.expected_final_states)
    return float(np.max(out_err)) if out_err.size else 0.0, \
           float(np.max(final_err)) if final_err.size else 0.0, \
           float(np.max(np.abs(case.expected_outputs[live]))) if np.any(live) else 1.0, \
           float(np.max(np.abs(case.expected_final_states)))


def expanded_equation_check():
    keys=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],dtype=np.float64)
    values=np.array([[1.,-2.],[-3.,4.],[5.,-6.],[-7.,8.]],dtype=np.float64)
    queries=np.array([[1.,2.],[-1.,3.],[4.,-2.],[3.,1.]],dtype=np.float64)
    alpha=np.array([.9,.8,.7,.6],dtype=np.float64)
    beta=np.array([.4,.3,.2,.1],dtype=np.float64)
    initial=np.array([[2.,-1.],[.5,3.]],dtype=np.float64)
    direct=initial.copy(); expanded=initial.copy(); maximum=0.0; ref_max=0.0
    eye=np.eye(2,dtype=np.float64)
    for t in range(len(alpha)):
        d=alpha[t]*direct
        e=values[t]-d@keys[t]
        direct=d+beta[t]*np.outer(e,keys[t])
        k=keys[t]
        expanded=alpha[t]*expanded@(eye-beta[t]*np.outer(k,k))+beta[t]*np.outer(values[t],k)
        maximum=max(maximum,float(np.max(np.abs(direct-expanded))))
        ref_max=max(ref_max,float(np.max(np.abs(direct@queries[t]))))
    return maximum,ref_max


def main():
    bundle_path=Path("recurrence_inputs.bin")
    cases,bundle_meta=decode_bundle(bundle_path.read_bytes())
    shape_list=[{"case_id":c.case_id,"T":c.shape[0],"B":c.shape[1],
                 "d_v":c.shape[2],"d_k":c.shape[3],
                 "live_steps":int(np.sum(c.lengths))} for c in cases]
    numerical={
        "operator":"D=alpha*S_previous; e=v-D@k; S=D+beta*outer(e,k); o=S@q",
        "state_orientation":"S[d_v,d_k]",
        "output_state":"post-update",
        "layout":"time-major interleaved [T,B,d_k/d_v], state row-major [B,d_v,d_k]",
        "represented_inputs":"FP32 keys, values, queries, alpha, beta, and initial state; FP64 independent recurrence outputs/final states serialized in the same bundle",
        "accumulation":"FP32 native; scalar FP64 independent semantic oracle",
        "active_key_norm":"unit norm in represented FP32 bundle; reader verifies; native kernel does not normalize",
        "gate_range":"0 <= alpha,beta <= 1, includes exact zero and near-one values",
        "sequence_contract":"all live timesteps in order, preserve nonzero initial state, carry final state across segments, no history truncation or sequence mixing",
        "padding":"NaN input/reference sentinels only outside each sequence length; output sentinel must remain untouched",
        "exceptional_values":"active inputs and initial states finite; padding NaNs are excluded from live operation",
        "atol":ATOL,
        "rtol":RTOL,
        "comparison":"abs(actual-reference) <= atol + rtol*abs(reference) for every live output and every final-state element"
    }
    sig_obj={"case_id":CASE_ID,"bundle_sha256":bundle_meta["bundle_file_sha256"],
             "cases":shape_list,"numerical_contract":numerical}
    contract={"format":"KERNELWIKI-VOLTA-RECURRENCE-CONTRACT/1",
              "case_id":CASE_ID,"case_signature":canonical_sha(sig_obj),
              "numerical_contract":numerical,"subcases":shape_list,
              "represented_input":{"path":BUNDLE_PATH,"sha256":bundle_meta["bundle_file_sha256"]}}
    output_err=final_err=output_ref=final_ref=0.0
    for c in cases:
        oe,fe,orr,frr=segmented_check(c)
        output_err=max(output_err,oe); final_err=max(final_err,fe)
        output_ref=max(output_ref,orr); final_ref=max(final_ref,frr)
    expanded_err,expanded_ref=expanded_equation_check()
    oracle={
        "format":"KERNELWIKI-CPU-ORACLE/1","case_id":CASE_ID,"passed":True,
        "represented_input":{"path":BUNDLE_PATH,"sha256":bundle_meta["bundle_file_sha256"]},
        "expected_reference_generation":{
            "inputs":"read exact little-endian FP32 represented values from the hashed bundle",
            "outputs":"FP64 recurrence in time order; record every live output and final state",
            "source":"recurrence_oracle.py:run_interleaved and run_sequence",
            "padding":"padded locations are NaN; they are not recurrence steps"
        },
        "checks":[
            {"name":"expanded_rank_one_equation_small_semantic_case",
             "max_abs_error":expanded_err,"reference_max_abs":expanded_ref,
             "atol":1e-14,"rtol":1e-14},
            {"name":"segmented_carry_all_bundle_outputs",
             "max_abs_error":output_err,"reference_max_abs":output_ref,
             "atol":0.0,"rtol":0.0},
            {"name":"segmented_carry_all_bundle_final_states",
             "max_abs_error":final_err,"reference_max_abs":final_ref,
             "atol":0.0,"rtol":0.0}
        ]
    }
    oracle["passed"]=all(
        check["max_abs_error"] <= check["atol"]+check["rtol"]*check["reference_max_abs"]
        for check in oracle["checks"]
    )
    if not oracle["passed"]:
        raise SystemExit("CPU oracle self-check did not pass")
    CONTRACT_PATH.write_text(json.dumps(contract,indent=2,sort_keys=True)+"\n")
    ORACLE_PATH.write_text(json.dumps(oracle,indent=2,sort_keys=True)+"\n")
    contract_sha=hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    oracle_sha=hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest()
    template={
      "format":"KERNELWIKI-VOLTA-EVIDENCE/1",
      "status":"scratch_template_not_run",
      "cases":[{
        "id":CASE_ID,"disposition":"not_run","gpu_uuid":"",
        "contract":{"path":"contract.json","sha256":contract_sha},
        "represented_input":{"path":BUNDLE_PATH,"sha256":bundle_meta["bundle_file_sha256"]},
        "oracle":{"path":"cpu-oracle.json","sha256":oracle_sha},
        "implementations":{
          "baseline":{"source":{"path":"src/gdn_runner.cu","sha256":"UPDATE_ON_ADOPTION"},
                       "binary":{"path":"bin/gdn_runner","sha256":"UPDATE_AFTER_BUILD"},
                       "executed_path":"benchmark_runs/volta-recurrence/bin/gdn_runner"},
          "candidate":{"source":{"path":"src/gdn_runner.cu","sha256":"UPDATE_ON_ADOPTION"},
                        "binary":{"path":"bin/gdn_runner","sha256":"UPDATE_AFTER_BUILD"},
                        "executed_path":"benchmark_runs/volta-recurrence/bin/gdn_runner"}},
        "libraries":[],"runs":{}
      }]
    }
    SUMMARY_TEMPLATE_PATH.write_text(json.dumps(template,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"case_signature":contract["case_signature"],
                      "bundle_sha256":bundle_meta["bundle_file_sha256"],
                      "contract_sha256":contract_sha,"oracle_sha256":oracle_sha,
                      "oracle_checks":oracle["checks"],"oracle_passed":oracle["passed"]},indent=2))

if __name__=="__main__":
    main()
