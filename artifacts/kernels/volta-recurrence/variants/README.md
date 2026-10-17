# Volta gated-delta recurrence variants

This source implements one bounded stateful operator for measured SM70
representative shapes. The state is FP32 row-major `S[sequence,d_v,d_k]`; each
sequence is independent and the caller provides the initial state and its
already prepared inputs. For each live token, the update is

```text
D = alpha * S_previous
e = v - D @ k
S = D + beta * outer(e, k)
o = S @ q                 # read the post-update state
```

`gdn_recurrent_state.cuh` contains two complete mappings. `per_token` submits
one kernel per timestep; that launch boundary forces each step to load and
store its state rows through global memory. `retained` submits one fused kernel
for a sequence and keeps one complete `[d_k]` state row in each row-owning lane
until final-state writeback. These are a whole-operator mapping comparison;
any timing difference also reflects launch count, synchronization and device
resource use, so it does not isolate state traffic as a cause.

Each CTA owns one sequence. Each lane owns one output/state row. The retained
lane holds its entire `d_k` row in a private array; `-Xptxas=-v` output must be
reviewed for registers and spills before interpreting the long `d_k=64` case.
The complete `per_token` baseline is retained as a correctness and timing
fallback if the private rows spill or reduce useful residency.

Inputs are consumed without normalization or rounding. The prepared bundle has
represented unit keys and gates in `[0,1]`; that test restriction is not a
general numerical bound for unnormalized keys. The caller's contract owns
input preparation, and the implementation must preserve the exact sequence,
initial state, and every returned output. It carries the full state over each
segment and does not truncate history.

The host reader validates a hashed FP32-input / FP64-reference bundle before a
device run. Its fixed comparison is `abs(actual-reference) <= 2e-5 +
2e-4*abs(reference)`, checked element-by-element on live outputs and every
final-state value. CUDA source and native GPU evidence remain unverified until
the controller run has completed; source existence is not a correctness or
performance result.
