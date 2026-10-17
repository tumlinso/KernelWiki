# KWV-07 represented input and FP64 reference files

All arrays are little endian. Each record starts with an eight-byte magic and
eight `uint32` fields encoded as `<8s8I` (40 bytes). Input records use
`KWV07I01`; reference records use `KWV07R01`. `bundle-manifest.json` records
every file's length and SHA-256. `suites/*.json` and `suite-files.json` bind
the two operator families and exact member files.

## Grouped/gated inputs

The header fields after magic are kind `1`, input rows, `K`, `N`, expert count,
group size `0`, assignment count, and auxiliary count `0`. Payload order is
FP16 `X[rows,K]`, FP16 `W_up[experts,K,N]`, FP16
`W_gate[experts,K,N]`, then int32 triples `(row_id,expert_id,output_position)`.
Every assignment is retained. Output positions must be a permutation of the
assignment range; repeated `(row_id,expert_id)` pairs are legal.

Reference payload order is FP64 output `Y[M,N]`, FP64 projected `U[M,N]`,
FP64 projected `G[M,N]`, then FP64 per-output `sum(abs(X*W_up))` and
`sum(abs(X*W_gate))` arrays, all in output-position order.

## W4A16 inputs

The header fields after magic are kind `2`, M, K, N, expert count `0`,
group size, assignment count `0`, and scale count. Payload order is FP16
activation `[M,K]`, packed unsigned bytes for row-major signed two's-complement
weights `[K,N]`, and FP32 scales `[ceil(K/group_size),N]`. The even logical
element occupies the low nibble; the odd element occupies the high nibble. An
unused high nibble after an odd total element count is outside the logical
tensor. Codes are `[-8,7]`.

Reference payload order is FP64 output `[M,N]` and FP64 per-output
`sum(abs(activation*decoded_weight))`. The CPU generator reads exact represented
FP16 activations and FP32 scales, decodes `round_fp16_rn(round_fp32(scale*q))`,
and accumulates the resulting FP16 values in FP64.

## CPU-only suite wrappers

Each suite has a JSON contract and a JSON oracle identity/precheck record. The
contract binds the exact suite manifest and its per-case input/reference/
contract hashes. The oracle record verifies that FP64 reference bytes are
already present and runs a declared CPU FP32 policy precheck over the
represented inputs. Its `passed` field means only that CPU checks succeeded;
it carries no GPU run metadata or GPU error values. Native checked output
records are emitted by the runner and captured in controller evidence.
