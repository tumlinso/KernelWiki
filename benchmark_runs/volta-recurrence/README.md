# KWV-08 recurrence benchmark package

This staging package is prepared for the KWV-08 native state-ownership
comparison. It contains no native CUDA execution evidence. The represented
input bundle was copied byte-for-byte from the earlier scratch preparation;
it was not regenerated here.

## Operator and variants

The represented inputs are FP32 `k[t, seq, d_k]`, `v[t, seq, d_v]`,
`q[t, seq, d_k]`, scalar FP32 gates, and FP32 state `[seq, d_v, d_k]`.
At every live step the contract is:

```
D = alpha * S_previous
error = v - D @ k
S = D + beta * outer(error, k)
o = S @ q                 # post-update state
```

The operator does not normalize keys, round or otherwise transform caller
inputs, reset state at chunk boundaries, or truncate history. Test keys in the
prepared bundle are explicitly unit length; the native reader verifies their
represented norm and the kernel consumes them unchanged.

`gdn_recurrent_state.cuh` proposes two complete paths. `per_token` launches one
state-update kernel per token, reading and writing the full FP32 state each
step. `retained` launches once per sequence and keeps each lane's complete
contiguous `d_k` row in its private array for the full sequence. One CTA owns
one sequence; one lane owns one `d_v` row. Both return every live output and
final state. The full-time comparison includes the baseline's T launches and
host enqueue cost. Report host enqueue, host completion, and CUDA-event stream
interval separately: host submission gaps may leave the GPU idle inside the
event interval. This is a whole-operator mapping comparison, not isolated proof
that state traffic caused any difference. The retained row may spill as `d_k`
grows; inspect compiler register/spill data before interpreting results.

## Immutable input and independent oracle

`prepare_recurrence_bundle.py` writes `recurrence_inputs.bin` and the adjacent
JSON manifest. The binary holds three cases (decode, short prefill, long
prefill), all represented FP32 inputs, ragged lengths, nonzero initial states,
every FP64 output, and every FP64 final state. Active keys are generated as
unit vectors and are not changed by the native runner. All padded input slots
carry NaN sentinels; the runner initializes padded output slots to a finite
sentinel and checks that they remain untouched. The payload and complete file
have SHA-256 identities.

The native runner's `bundle_contract.hpp` validates the header, SHA-256,
shapes, live finiteness, gates, represented key norms, padding markers, and
zero-length state references before any GPU test. The references themselves
are generated independently in `recurrence_oracle.py` with FP64 arithmetic.
`test_recurrence_oracle.py` covers the formula, boundaries, long near-one
history, sequence order, segmentation/carry, and ragged sequence isolation.
`test_recurrence_bundle.py` checks binary round trips and rejects changed or
trailing bytes. `test_bundle_reader.cpp` exercises the same native host reader
and SHA implementation without CUDA.

The fixed comparison rule is `abs(actual-reference) <= 2e-5 +
2e-4*abs(reference)` for every live output and final-state element. The
CPU-only `fp32_policy_probe.json` emulates the runner's FP32 multiply/FMA order
and passed this rule for every short/decode sequence and eight long sequences,
including the near-one cases. The largest scaled error was 0.00473 of the
allowed tolerance; this is only a tolerance sanity check, and CUDA correctness
still needs actual native runs.

## Native result and checks

`gdn_runner.cu` accepts `--variant per_token|retained --mode
correctness|time|profile --bundle <path>`. Profile mode requires one explicit
`--case`; it checks every output and final-state element for that case, then
uses one full-sequence launch for NCU. It reports raw `cudaDeviceProp.uuid` hex,
compute capability, GPU name, both bundle hashes, and per-case checked,
failed, non-finite, maximum absolute and scaled errors. `status: passed` is
emitted only after every live output, final-state value, segmented-carry output,
segmented final-state value, and padded-output sentinel passes. A failed or
unavailable check retains its own counts and reason; it cannot be represented
as `passed`.

Every native run repeats the same sequence as one call and as segments split
at available boundaries near 1, 33, and 129 tokens. It carries the returned
state between segments and checks all outputs and final state against the same
FP64 oracle. The long case includes nonzero initial state, exact alpha/beta
zero gates, repeated and orthogonal keys, a full near-one gate sequence, and
ragged interleaving. The decode case has only one available token; it never
batches future inputs.

Time mode first completes every correctness/carry check. It measures seven
samples after two warmups, resetting state outside each sample. Host enqueue
covers dispatch of the full operator, host completion covers dispatch through
stream completion, and the CUDA-event interval is retained as a separate
measurement that may include device idle time while the host submits many
baseline launches. Allocations and input transfers are outside the timed
region. All host/device copies and kernels use one explicit nonblocking stream;
pageable host buffers remain alive until that stream is synchronized, including
exception cleanup. Reset copies synchronize before the measured interval.
A future native evidence record must bind compiler, executable, source,
bundle file, controller receipts, and normal unprofiled timing. Sanitizer and
selected profile evidence remain separate records.

## Build and controller use after claim

The draft is one translation unit because `gdn_runner.cu` includes the
header-only kernel source:

```sh
nvcc -O3 -std=c++17 -lineinfo -arch=sm_70 -Xptxas=-v \
  -o <repo>/benchmark_runs/volta-recurrence/bin/gdn_runner \
  <repo>/artifacts/kernels/volta-recurrence/variants/gdn_runner.cu
```

The `-Xptxas=-v` report is needed to check registers and spills for the
retained rows. The six controller roles compare the same immutable bundle and
one executable: per-token baseline correctness (and the sole build), retained
candidate correctness, retained candidate Compute Sanitizer, baseline timing,
candidate timing, and one focused retained-state profile on the long prefill
case. Every later role uses the exact first-build ELF. No clock or power
settings are part of this proposal.

## Scope boundary

This proposal does not implement a chunked recurrence. Chunking remains
deferred unless the claimed implementation supplies a cited equivalent
derivation, proves that the caller already has each full chunk's inputs, and
measures preparation, correction, conversion, and carry costs. The algorithm
source does not establish that its released kernels ran on SM70.

No CUDA source in this directory has been compiled or run. The C++ host bundle
reader and the Python semantic suite have only been built or run on the CPU.
